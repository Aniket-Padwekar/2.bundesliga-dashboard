"""
2. Bundesliga Dashboard — Data & Predictions Pipeline (model v2)
================================================================
Runs automatically (GitHub Actions, daily). Every run, from scratch:
  1. Pulls every season's matches from OpenLigaDB (4 requests, no key needed)
  2. Builds Elo ratings (displayed on the dashboard; NOT used for predictions)
  3. Fits attack/defense strength per club, shrunk toward average for clubs
     with few games
  4. Simulates the rest of the season 10,000 times by sampling actual
     scorelines from a Poisson model (so goal difference and goals scored
     tie-breaks are simulated too), seeded from the data so identical
     results always give identical output
  5. Builds everything the dashboard and the promotion report need
  6. Keeps data/history.json: one model snapshot per completed matchday
  7. Writes data/dashboard_data.json
"""

import csv
import glob
import hashlib
import io
import json
import math
import os
import re
import time
from datetime import datetime, timezone

import numpy as np
import requests
from scipy.stats import poisson

# ── CONFIG ──────────────────────────────────────────────────────────────────
SEASONS = [2023, 2024, 2025, 2026]          # 2023 = 2023/24 season, etc.
LEAGUE = "bl2"                               # 2. Bundesliga on OpenLigaDB
CURRENT_SEASON = "2026/27"
K_FACTOR = 20                                # Elo sensitivity to each result
HOME_ADVANTAGE_ELO = 60                      # Elo points of home advantage
SEASON_REGRESSION = 0.75                     # ratings pulled 25% toward average each new season
SHRINKAGE_WEIGHT = 10                        # weight of the league-average prior, in matches
HOME_GOAL_BOOST = 1.12                       # home teams score ~12% more on average
N_SIMULATIONS = 10000
MODEL_VERSION = "v2"                         # bump when model logic changes: history is rebuilt
HISTORY_FILE = "data/history.json"
OUTPUT_FILE = "data/dashboard_data.json"
CONCEDED_GOALS_FILE = "data/conceded_goals.csv"      # input: goals conceded, with shot coordinates
SET_PIECE_SITUATIONS = {"corner", "set-piece", "throw-in-set-piece", "free-kick"}

# Sofascore shot coordinates are 0-100 along the pitch (x, measured from the goal line being
# attacked) and across it (y, 50 = centre). The scale is undocumented. Every penalty in the
# data sits at exactly x = 11.5, so x is calibrated to put the penalty spot at 11 m; y assumes
# a 68 m wide pitch. Distances derived from these are approximate.
PITCH = {"x_m_per_unit": round(11.0 / 11.5, 4), "y_m_per_unit": 0.68}


def season_label(year):
    return f"{year}/{str(year + 1)[-2:]}"


# ── STEP 1: PULL ALL MATCH DATA ──────────────────────────────────────────────
def fetch_all_matches():
    """Pulls each season's ENTIRE match list in a single request. OpenLigaDB
    enforces a shared rate limit (1000 requests/hour per IP) and GitHub
    Actions runners share IP pools, so 4 requests per run is deliberate."""
    all_matches = []

    for season in SEASONS:
        label = season_label(season)
        url = f"https://api.openligadb.de/getmatchdata/{LEAGUE}/{season}"
        matches = None

        for attempt in range(3):
            try:
                resp = requests.get(url, timeout=20)
                if resp.status_code == 200:
                    matches = resp.json()
                    break
                print(f"  WARNING: season {label} returned HTTP {resp.status_code} (attempt {attempt+1})")
            except requests.RequestException as e:
                print(f"  WARNING: season {label} request failed: {e} (attempt {attempt+1})")
            time.sleep(2)

        time.sleep(1)

        if not matches:
            print(f"  ERROR: season {label} — could not fetch after 3 attempts. Skipping this season entirely.")
            continue

        unscored, malformed, finished_count = 0, 0, 0
        for m in matches:
            home = (m.get("team1") or {}).get("teamName") or ""
            away = (m.get("team2") or {}).get("teamName") or ""
            if not home or not away:
                malformed += 1
                continue
            matchday = (m.get("group") or {}).get("groupOrderID") or 0

            results = m.get("matchResults") or []
            final = next((r for r in results if r.get("resultTypeID") == 2), None) or (results[-1] if results else None)
            gh = final.get("pointsTeam1") if final else None
            ga = final.get("pointsTeam2") if final else None
            has_score = gh is not None and ga is not None

            # The feed is community-entered: a match can be flagged finished before its score is in.
            # Such a match is treated as not yet played until the score appears.
            flagged = bool(m.get("matchIsFinished", False))
            if flagged and not has_score:
                unscored += 1
            is_finished = flagged and has_score
            finished_count += 1 if is_finished else 0

            date_str = m.get("matchDateTime", "")[:10] if m.get("matchDateTime") else ""
            all_matches.append({
                "season": label, "matchday": matchday, "date": date_str,
                "home": home, "away": away,
                "home_goals": int(gh) if is_finished else None,
                "away_goals": int(ga) if is_finished else None,
                "finished": is_finished
            })

        print(f"  {label}: {len(matches)} total matches fetched, {finished_count} finished")
        if unscored:
            print(f"  NOTE: {label}: {unscored} match(es) flagged finished but without a score yet, treated as not yet played")
        if malformed:
            print(f"  NOTE: {label}: skipped {malformed} malformed match record(s)")

    return all_matches


# ── SHARED HELPERS ───────────────────────────────────────────────────────────
def rank_table(season_matches):
    """Final/current table in official 2. Bundesliga order: points, goal
    difference, goals scored (then name, only to make ties deterministic;
    head-to-head is not implemented)."""
    rows = {}
    for m in season_matches:
        if not m["finished"] or m["home_goals"] is None:
            continue
        h, a, hg, ag = m["home"], m["away"], m["home_goals"], m["away_goals"]
        for t in (h, a):
            rows.setdefault(t, {"team": t, "played": 0, "points": 0, "gf": 0, "ga": 0})
        rows[h]["played"] += 1
        rows[a]["played"] += 1
        rows[h]["gf"] += hg; rows[h]["ga"] += ag
        rows[a]["gf"] += ag; rows[a]["ga"] += hg
        if hg > ag:
            rows[h]["points"] += 3
        elif hg < ag:
            rows[a]["points"] += 3
        else:
            rows[h]["points"] += 1
            rows[a]["points"] += 1

    table = sorted(rows.values(), key=lambda r: (-r["points"], -(r["gf"] - r["ga"]), -r["gf"], r["team"]))
    for i, r in enumerate(table, start=1):
        r["position"] = i
        r["gd"] = r["gf"] - r["ga"]
    return table


def outcome_for_position(pos, n_teams=18):
    if pos <= 2:
        return "promoted"
    if pos == 3:
        return "playoff"
    if pos == n_teams - 2:      # 16th
        return "relegation_playoff"
    if pos >= n_teams - 1:      # 17th, 18th
        return "relegated"
    return "midtable"


# ── STEP 2: ELO (display only) ───────────────────────────────────────────────
def expected_score(ra, rb):
    return 1.0 / (1.0 + 10 ** ((rb - ra) / 400))


def build_elo_ratings(played_matches, snapshot_season=None):
    """Chronological Elo. Returns (final ratings, ratings at the start of
    snapshot_season). New teams start at 1500; ratings regress toward the
    mean between seasons. NOTE: shown on the Promotion report (current
    rating, and the change since the season started) purely for context —
    it is not an input to the fixture probabilities or any prediction."""
    ratings = {}
    start_ratings = None
    current_season = None

    for m in sorted(played_matches, key=lambda m: (m["season"], m["date"])):
        season, home, away = m["season"], m["home"], m["away"]
        hg, ag = m["home_goals"], m["away_goals"]
        if hg is None or ag is None:
            continue

        ratings.setdefault(home, 1500.0)
        ratings.setdefault(away, 1500.0)

        if season != current_season:
            if current_season is not None:
                for t in ratings:
                    ratings[t] = 1500 + (ratings[t] - 1500) * SEASON_REGRESSION
            current_season = season
            if snapshot_season is not None and season == snapshot_season:
                start_ratings = dict(ratings)

        ra, rb = ratings[home], ratings[away]
        exp_home = expected_score(ra + HOME_ADVANTAGE_ELO, rb)
        result_home = 1.0 if hg > ag else (0.5 if hg == ag else 0.0)

        ratings[home] = ra + K_FACTOR * (result_home - exp_home)
        ratings[away] = rb + K_FACTOR * ((1 - result_home) - (1 - exp_home))

    if start_ratings is None:
        start_ratings = dict(ratings)
    return ratings, start_ratings


# ── STEP 3: ATTACK / DEFENSE, WITH SHRINKAGE ─────────────────────────────────
def build_attack_defense(played_matches, current_teams):
    """Last season + this season so far, goals for/against per game relative
    to the league average, shrunk toward 1.0 for clubs with few games."""
    recent = [m for m in played_matches
              if m["season"] in [season_label(SEASONS[-2]), season_label(SEASONS[-1])]
              and m["home_goals"] is not None]

    goals_scored = {t: [] for t in current_teams}
    goals_conceded = {t: [] for t in current_teams}

    for m in recent:
        h, a, hg, ag = m["home"], m["away"], m["home_goals"], m["away_goals"]
        if h in current_teams:
            goals_scored[h].append(hg)
            goals_conceded[h].append(ag)
        if a in current_teams:
            goals_scored[a].append(ag)
            goals_conceded[a].append(hg)

    all_goals = [g for vals in goals_scored.values() for g in vals]
    league_avg = float(np.mean(all_goals)) if all_goals else 1.5

    attack, defense = {}, {}
    for t in current_teams:
        n = len(goals_scored[t])
        if n == 0:
            attack[t], defense[t] = 1.0, 1.0
            continue
        raw_attack = np.mean(goals_scored[t]) / league_avg
        raw_defense = np.mean(goals_conceded[t]) / league_avg
        attack[t] = float((n * raw_attack + SHRINKAGE_WEIGHT * 1.0) / (n + SHRINKAGE_WEIGHT))
        defense[t] = float((n * raw_defense + SHRINKAGE_WEIGHT * 1.0) / (n + SHRINKAGE_WEIGHT))

    return attack, defense, league_avg


# ── TEAM COLORS (hard-coded: Sofascore blocks GitHub Actions IPs) ────────────
DEFAULT_COLORS = {"primary": "#1a1a2e", "secondary": "#e0e0e0", "text": "#ffffff"}

TEAM_COLORS = {
    "Hertha BSC":              {"primary": "#005CA9", "secondary": "#FFFFFF", "text": "#FFFFFF"},
    "1. FC Nürnberg":          {"primary": "#C8102E", "secondary": "#000000", "text": "#FFFFFF"},
    "VfL Wolfsburg":           {"primary": "#65B32E", "secondary": "#FFFFFF", "text": "#FFFFFF"},
    "1. FC Heidenheim 1846":   {"primary": "#E2001A", "secondary": "#003B79", "text": "#FFFFFF"},
    "1. FC Kaiserslautern":    {"primary": "#E2001A", "secondary": "#FFFFFF", "text": "#FFFFFF"},
    "Hannover 96":             {"primary": "#006633", "secondary": "#000000", "text": "#FFFFFF"},
    "Energie Cottbus":         {"primary": "#E2001A", "secondary": "#FFFFFF", "text": "#FFFFFF"},
    "1. FC Magdeburg":         {"primary": "#0068B2", "secondary": "#FFFFFF", "text": "#FFFFFF"},
    "FC St. Pauli":            {"primary": "#EC1B24", "secondary": "#000000", "text": "#FFFFFF"},
    "VfL Bochum":              {"primary": "#004B9B", "secondary": "#FFFFFF", "text": "#FFFFFF"},
    "VfL Osnabrück":           {"primary": "#6A0DAD", "secondary": "#FFFFFF", "text": "#FFFFFF"},
    "DSC Arminia Bielefeld":   {"primary": "#003C7D", "secondary": "#000000", "text": "#FFFFFF"},
    "SV Darmstadt 98":         {"primary": "#003C7D", "secondary": "#FFFFFF", "text": "#FFFFFF"},
    "Dynamo Dresden":          {"primary": "#FFD500", "secondary": "#000000", "text": "#000000"},
    "SpVgg Greuther Fürth":    {"primary": "#00873E", "secondary": "#FFFFFF", "text": "#FFFFFF"},
    "Holstein Kiel":           {"primary": "#E2001A", "secondary": "#003087", "text": "#FFFFFF"},
    "Karlsruher SC":           {"primary": "#002E62", "secondary": "#FFFFFF", "text": "#FFFFFF"},
    "Eintracht Braunschweig":  {"primary": "#FFA000", "secondary": "#003C7D", "text": "#000000"},
    "FC Schalke 04":           {"primary": "#004B9B", "secondary": "#FFFFFF", "text": "#FFFFFF"},
    "SV 07 Elversberg":        {"primary": "#000000", "secondary": "#FFFFFF", "text": "#FFFFFF"},
    "SC Paderborn 07":         {"primary": "#003C7D", "secondary": "#FFFFFF", "text": "#FFFFFF"},
    "Hamburger SV":            {"primary": "#005CA9", "secondary": "#000000", "text": "#FFFFFF"},
    "1. FC Köln":              {"primary": "#E2001A", "secondary": "#FFFFFF", "text": "#FFFFFF"},
    "Hansa Rostock":           {"primary": "#003C7D", "secondary": "#FFFFFF", "text": "#FFFFFF"},
    "SV Wehen Wiesbaden":      {"primary": "#E2001A", "secondary": "#000000", "text": "#FFFFFF"},
    "Jahn Regensburg":         {"primary": "#003C7D", "secondary": "#FFFFFF", "text": "#FFFFFF"},
    "Fortuna Düsseldorf":      {"primary": "#E2001A", "secondary": "#FFFFFF", "text": "#FFFFFF"},
    "Preußen Münster":         {"primary": "#000000", "secondary": "#FFFFFF", "text": "#FFFFFF"},
    "SSV Ulm 1846":            {"primary": "#003C7D", "secondary": "#FFFFFF", "text": "#FFFFFF"},
}


def fetch_team_colors(team_names):
    colors, missing = {}, []
    for name in team_names:
        if name in TEAM_COLORS:
            colors[name] = TEAM_COLORS[name]
        else:
            colors[name] = DEFAULT_COLORS
            missing.append(name)
    if missing:
        print(f"  NOTE: no color entry for {len(missing)} team(s) — used fallback: {missing}")
    return colors


# ── STEP 4: FIXTURE PROBABILITIES AND SEASON SIMULATION ──────────────────────
def expected_goals(home, away, attack, defense, league_avg):
    eg_home = league_avg * attack[home] * defense[away] * HOME_GOAL_BOOST
    eg_away = league_avg * attack[away] * defense[home]
    return eg_home, eg_away


def match_outcome_probs(eg_home, eg_away, max_goals=10):
    """Win/draw/loss probabilities from independent Poisson goal counts."""
    goals = np.arange(max_goals + 1)
    home_p = poisson.pmf(goals, eg_home)
    away_p = poisson.pmf(goals, eg_away)
    grid = np.outer(home_p, away_p)
    hw = float(np.tril(grid, -1).sum())     # home goals > away goals
    dr = float(np.trace(grid))
    aw = float(np.triu(grid, 1).sum())
    total = hw + dr + aw
    return hw / total, dr / total, aw / total


def data_seed(played_matches):
    """Seed derived from the finished results themselves: identical data
    always produces identical simulations, new results change the seed."""
    parts = sorted(f"{m['season']}|{m['matchday']}|{m['home']}|{m['away']}|{m['home_goals']}|{m['away_goals']}"
                   for m in played_matches)
    digest = hashlib.sha256(("\n".join(parts) + MODEL_VERSION).encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def run_monte_carlo(teams, stats, fixtures, n_sim, seed):
    """Simulates the remaining fixtures n_sim times, sampling actual goals
    from each fixture's Poisson expectation. Final order uses official
    tie-breaks: points, goal difference, goals scored (random only if all
    three are equal). Returns (position_counts[team_idx][finish_idx],
    final_points[sim][team_idx])."""
    rng = np.random.default_rng(seed)
    T, F = len(teams), len(fixtures)
    idx = {t: i for i, t in enumerate(teams)}

    pts0 = np.array([stats[t]["points"] for t in teams], dtype=np.int64)
    gf0 = np.array([stats[t]["gf"] for t in teams], dtype=np.int64)
    gd0 = np.array([stats[t]["gf"] - stats[t]["ga"] for t in teams], dtype=np.int64)

    sim_pts = np.tile(pts0, (n_sim, 1))
    sim_gf = np.tile(gf0, (n_sim, 1))
    sim_gd = np.tile(gd0, (n_sim, 1))

    if F > 0:
        hi = np.array([idx[f["home"]] for f in fixtures])
        ai = np.array([idx[f["away"]] for f in fixtures])
        lam_h = np.array([f["eg_home"] for f in fixtures])
        lam_a = np.array([f["eg_away"] for f in fixtures])

        gh = rng.poisson(lam_h, size=(n_sim, F)).astype(np.int64)
        ga = rng.poisson(lam_a, size=(n_sim, F)).astype(np.int64)
        home_pts = 3 * (gh > ga).astype(np.int64) + (gh == ga).astype(np.int64)
        away_pts = 3 * (gh < ga).astype(np.int64) + (gh == ga).astype(np.int64)
        diff = gh - ga

        for t in range(T):
            hm, am = hi == t, ai == t
            sim_pts[:, t] += home_pts[:, hm].sum(axis=1) + away_pts[:, am].sum(axis=1)
            sim_gf[:, t] += gh[:, hm].sum(axis=1) + ga[:, am].sum(axis=1)
            sim_gd[:, t] += diff[:, hm].sum(axis=1) - diff[:, am].sum(axis=1)

    tiebreak = rng.integers(0, 10, size=(n_sim, T)).astype(np.int64)
    key = sim_pts * 10**9 + (sim_gd + 1000) * 10**5 + sim_gf * 10 + tiebreak
    order = np.argsort(-key, axis=1)            # order[s, k] = team finishing k+1 in sim s

    position_counts = np.zeros((T, T), dtype=np.int64)
    for k in range(T):
        position_counts[:, k] = np.bincount(order[:, k], minlength=T)

    return position_counts, sim_pts


def current_stats(teams, played_this_season):
    stats = {t: {"played": 0, "points": 0, "gf": 0, "ga": 0, "w": 0, "d": 0, "l": 0} for t in teams}
    for m in played_this_season:
        h, a, hg, ag = m["home"], m["away"], int(m["home_goals"]), int(m["away_goals"])
        for t in (h, a):
            stats[t]["played"] += 1
        stats[h]["gf"] += hg; stats[h]["ga"] += ag
        stats[a]["gf"] += ag; stats[a]["ga"] += hg
        if hg > ag:
            stats[h]["points"] += 3; stats[h]["w"] += 1; stats[a]["l"] += 1
        elif hg < ag:
            stats[a]["points"] += 3; stats[a]["w"] += 1; stats[h]["l"] += 1
        else:
            stats[h]["points"] += 1; stats[a]["points"] += 1
            stats[h]["d"] += 1; stats[a]["d"] += 1
    return stats


def prepare_matches(all_matches, as_of_matchday):
    """View of the data as it stood after a given matchday: later matches of
    the current season are treated as not yet played."""
    if as_of_matchday is None:
        return all_matches
    out = []
    for m in all_matches:
        if m["season"] == CURRENT_SEASON and m["matchday"] > as_of_matchday and m["finished"]:
            m2 = dict(m)
            m2["finished"] = False
            m2["home_goals"] = None
            m2["away_goals"] = None
            out.append(m2)
        else:
            out.append(m)
    return out


def model_snapshot(matches, n_sim=N_SIMULATIONS):
    """One full model run on the given view of the data."""
    played = [m for m in matches if m["finished"]]
    season_matches = [m for m in matches if m["season"] == CURRENT_SEASON]
    teams = sorted({m["home"] for m in season_matches} | {m["away"] for m in season_matches})
    upcoming = sorted([m for m in season_matches if not m["finished"]],
                      key=lambda m: (m["matchday"], m["home"]))

    elo, elo_start = build_elo_ratings(played, snapshot_season=CURRENT_SEASON)
    attack, defense, league_avg = build_attack_defense(played, teams)
    played_now = [m for m in played if m["season"] == CURRENT_SEASON]
    stats = current_stats(teams, played_now)

    fixtures = []
    for m in upcoming:
        egh, ega = expected_goals(m["home"], m["away"], attack, defense, league_avg)
        hw, dr, aw = match_outcome_probs(egh, ega)

        # Schedule difficulty as a league-average club (attack = defense = 1.0) would
        # experience this fixture. Independent of the club's own strength and results,
        # so strong clubs are not flattered and weak clubs are not punished.
        avg_hw, avg_dr, _ = match_outcome_probs(league