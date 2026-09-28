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

import hashlib
import json
import os
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

        for m in matches:
            is_finished = m.get("matchIsFinished", False)
            home = m.get("team1", {}).get("teamName", "")
            away = m.get("team2", {}).get("teamName", "")
            matchday = m.get("group", {}).get("groupOrderID", 0)

            results = m.get("matchResults", [])
            final = next((r for r in results if r.get("resultTypeID") == 2), None) or (results[-1] if results else None)

            home_goals = int(final.get("pointsTeam1")) if (is_finished and final) else None
            away_goals = int(final.get("pointsTeam2")) if (is_finished and final) else None
            date_str = m.get("matchDateTime", "")[:10] if m.get("matchDateTime") else ""

            all_matches.append({
                "season": label, "matchday": matchday, "date": date_str,
                "home": home, "away": away,
                "home_goals": home_goals, "away_goals": away_goals,
                "finished": bool(is_finished)
            })

        finished_count = sum(1 for m in matches if m.get("matchIsFinished"))
        print(f"  {label}: {len(matches)} total matches fetched, {finished_count} finished")

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
    mean between seasons. NOTE: Elo is shown on the dashboard but is not an
    input to the fixture probabilities."""
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
        avg_hw, avg_dr, _ = match_outcome_probs(league_avg * defense[m["away"]] * HOME_GOAL_BOOST,
                                                league_avg * attack[m["away"]])
        _, avg_dr2, avg_aw2 = match_outcome_probs(league_avg * attack[m["home"]] * HOME_GOAL_BOOST,
                                                  league_avg * defense[m["home"]])
        fixtures.append({"matchday": m["matchday"], "home": m["home"], "away": m["away"],
                         "eg_home": egh, "eg_away": ega, "hw": hw, "dr": dr, "aw": aw,
                         "avg_ep_home": 3 * avg_hw + avg_dr,      # average club playing this opponent at home
                         "avg_ep_away": 3 * avg_aw2 + avg_dr2})   # average club playing this opponent away

    counts, sim_pts = run_monte_carlo(teams, stats, fixtures, n_sim, data_seed(played))

    return {"teams": teams, "stats": stats, "attack": attack, "defense": defense,
            "league_avg": league_avg, "elo": elo, "elo_start": elo_start,
            "fixtures": fixtures, "counts": counts, "sim_pts": sim_pts,
            "n_sim": n_sim, "played_now": played_now}


# ── OUTPUT BUILDERS ──────────────────────────────────────────────────────────
def build_predicted_table(state):
    teams, stats, counts, n_sim = state["teams"], state["stats"], state["counts"], state["n_sim"]
    rows = []
    for i, t in enumerate(teams):
        dist = counts[i] / n_sim
        positions = np.arange(1, len(teams) + 1)
        pts = state["sim_pts"][:, i]
        p10, p50, p90 = np.percentile(pts, [10, 50, 90])
        rows.append({
            "team": t,
            "current_points": stats[t]["points"],
            "current_gd": stats[t]["gf"] - stats[t]["ga"],
            "current_gf": stats[t]["gf"],
            "current_ga": stats[t]["ga"],
            "matches_played": stats[t]["played"],
            "wins": stats[t]["w"], "draws": stats[t]["d"], "losses": stats[t]["l"],
            "avg_projected_position": round(float((dist * positions).sum()), 2),
            "promotion_probability": round(float(dist[:2].sum()), 4),
            "champion_probability": round(float(dist[0]), 4),
            "playoff_probability": round(float(dist[2]), 4),
            "position_distribution": [round(float(p), 4) for p in dist],
            "projected_points": {"mean": round(float(pts.mean()), 1),
                                 "p10": int(round(p10)), "p50": int(round(p50)), "p90": int(round(p90))},
            "elo_rating": round(state["elo"].get(t, 1500), 1),
            "elo_start_of_season": round(state["elo_start"].get(t, 1500), 1),
            "attack_rating": round(state["attack"][t], 3),
            "defense_rating": round(state["defense"][t], 3),
        })
    rows.sort(key=lambda r: r["avg_projected_position"])
    return rows


def build_trajectories(all_matches, seasons_wanted):
    """Cumulative points by game, for every team, for the requested seasons."""
    trajectories = {}
    for label in seasons_wanted:
        season_matches = sorted([m for m in all_matches if m["season"] == label and m["finished"]],
                                key=lambda m: m["matchday"])
        team_points, team_traj = {}, {}
        for m in season_matches:
            h, a, hg, ag = m["home"], m["away"], m["home_goals"], m["away_goals"]
            for t in (h, a):
                if t not in team_points:
                    team_points[t] = 0
                    team_traj[t] = []
            if hg > ag:
                team_points[h] += 3
            elif hg < ag:
                team_points[a] += 3
            else:
                team_points[h] += 1
                team_points[a] += 1
            team_traj[h].append({"matchday": m["matchday"], "points": team_points[h]})
            team_traj[a].append({"matchday": m["matchday"], "points": team_points[a]})
        trajectories[label] = team_traj
    return trajectories


def build_promotion_benchmarks(all_matches, labels):
    """Champion, runner-up (the automatic-promotion cut-off) and third place
    of each completed season, ranked with official tie-breaks."""
    benchmarks = {}
    for label in labels:
        season_matches = [m for m in all_matches if m["season"] == label and m["finished"]]
        table = rank_table(season_matches)
        if len(table) < 3:
            continue
        season_matches.sort(key=lambda m: m["matchday"])

        def trajectory_for(team):
            pts, traj = 0, []
            for m in season_matches:
                h, a, hg, ag = m["home"], m["away"], m["home_goals"], m["away_goals"]
                if team not in (h, a):
                    continue
                if (h == team and hg > ag) or (a == team and ag > hg):
                    pts += 3
                elif hg == ag:
                    pts += 1
                traj.append({"matchday": m["matchday"], "points": pts})
            return traj

        champ, second, third = table[0], table[1], table[2]
        safe = table[len(table) - 4] if len(table) >= 8 else None    # 15th of 18: last place clear of the relegation play-off
        benchmarks[label] = {
            "champion": {"team": champ["team"], "final_points": champ["points"],
                         "trajectory": trajectory_for(champ["team"])},
            "promoted_2nd": {"team": second["team"], "final_points": second["points"],
                             "final_gd": second["gd"], "trajectory": trajectory_for(second["team"])},
            "third_place": {"team": third["team"], "final_points": third["points"]},
        }
        if safe:
            benchmarks[label]["last_safe_place"] = {"team": safe["team"], "final_points": safe["points"],
                                                    "position": safe["position"]}
    return benchmarks


def build_fixture_difficulty(fixtures):
    """Per-team upcoming fixtures with the model's win/draw probabilities,
    expected points and a 1-5 difficulty rating from the club's own win
    probability."""
    def difficulty(p):
        if p >= 0.55: return 1
        if p >= 0.42: return 2
        if p >= 0.30: return 3
        if p >= 0.20: return 4
        return 5

    grid = {}
    for f in fixtures:
        grid.setdefault(f["home"], []).append({
            "matchday": f["matchday"], "opponent": f["away"], "is_home": True,
            "win_probability": round(f["hw"], 3), "draw_probability": round(f["dr"], 3),
            "expected_points": round(3 * f["hw"] + f["dr"], 3), "difficulty": difficulty(f["hw"]),
            "schedule_ep": round(f["avg_ep_home"], 3)})
        grid.setdefault(f["away"], []).append({
            "matchday": f["matchday"], "opponent": f["home"], "is_home": False,
            "win_probability": round(f["aw"], 3), "draw_probability": round(f["dr"], 3),
            "expected_points": round(3 * f["aw"] + f["dr"], 3), "difficulty": difficulty(f["aw"]),
            "schedule_ep": round(f["avg_ep_away"], 3)})
    for team in grid:
        grid[team].sort(key=lambda x: x["matchday"])
    return grid


def build_last_season_comparison(all_matches, state):
    """Each current club's record over its first N games last season (N =
    games played this season), plus its final position and points."""
    prev_label = season_label(SEASONS[-2])
    prev_matches = [m for m in all_matches if m["season"] == prev_label and m["finished"]]
    prev_table = {r["team"]: r for r in rank_table(prev_matches)}
    out = {}
    for t in state["teams"]:
        if t not in prev_table:
            out[t] = {"in_league_last_season": False}
            continue
        n = state["stats"][t]["played"]
        games = sorted([m for m in prev_matches if t in (m["home"], m["away"])], key=lambda m: m["matchday"])[:n]
        pts = gf = ga = 0
        for m in games:
            own, opp = (m["home_goals"], m["away_goals"]) if m["home"] == t else (m["away_goals"], m["home_goals"])
            gf += own; ga += opp
            pts += 3 if own > opp else (1 if own == opp else 0)
        out[t] = {"in_league_last_season": True, "matches": len(games),
                  "points_then": pts, "gf_then": gf, "ga_then": ga,
                  "final_position_last_season": prev_table[t]["position"],
                  "final_points_last_season": prev_table[t]["points"]}
    return out


def build_historical_analogues(all_matches, historical_labels, n_games):
    """Every club-season in the completed seasons: points after its first
    N games, and where it finished."""
    if n_games <= 0:
        return {"matches_played": 0, "rows": []}
    rows = []
    for label in historical_labels:
        season_matches = [m for m in all_matches if m["season"] == label and m["finished"]]
        table = rank_table(season_matches)
        n_teams = len(table)
        for r in table:
            t = r["team"]
            games = sorted([m for m in season_matches if t in (m["home"], m["away"])],
                           key=lambda m: m["matchday"])[:n_games]
            pts = 0
            for m in games:
                own, opp = (m["home_goals"], m["away_goals"]) if m["home"] == t else (m["away_goals"], m["home_goals"])
                pts += 3 if own > opp else (1 if own == opp else 0)
            rows.append({"season": label, "team": t, "points_after_n": pts,
                         "final_position": r["position"], "final_points": r["points"],
                         "outcome": outcome_for_position(r["position"], n_teams)})
    return {"matches_played": n_games, "rows": rows}


def build_recent_results(state, k=5):
    out = {t: [] for t in state["teams"]}
    for m in sorted(state["played_now"], key=lambda m: (m["matchday"], m["date"])):
        h, a, hg, ag = m["home"], m["away"], m["home_goals"], m["away_goals"]
        out[h].append({"matchday": m["matchday"], "opponent": a, "is_home": True, "gf": hg, "ga": ag,
                       "result": "W" if hg > ag else ("D" if hg == ag else "L")})
        out[a].append({"matchday": m["matchday"], "opponent": h, "is_home": False, "gf": ag, "ga": hg,
                       "result": "W" if ag > hg else ("D" if hg == ag else "L")})
    return {t: rows[-k:] for t, rows in out.items()}


# ── HISTORY: one snapshot per completed matchday, self-healing ───────────────
def load_history():
    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return []


def completed_matchdays(all_matches):
    by_md = {}
    for m in all_matches:
        if m["season"] == CURRENT_SEASON:
            by_md.setdefault(m["matchday"], []).append(m)
    return sorted(md for md, ms in by_md.items() if all(x["finished"] for x in ms))


def update_history(all_matches, existing):
    """Adds a snapshot for every completed matchday that doesn't have one,
    each computed only from results through that matchday. Snapshots made
    by a different MODEL_VERSION are discarded and rebuilt, so history never
    mixes old and new model logic."""
    history = [s for s in existing if s.get("model_version") == MODEL_VERSION]
    have = {s["matchday"] for s in history}

    for md in completed_matchdays(all_matches):
        if md in have:
            continue
        print(f"  Building history snapshot for matchday {md}...")
        snap = model_snapshot(prepare_matches(all_matches, md))
        table = build_predicted_table(snap)
        history.append({
            "matchday": md, "model_version": MODEL_VERSION, "n_simulations": snap["n_sim"],
            "teams": {r["team"]: {"points": r["current_points"], "gd": r["current_gd"],
                                  "promotion_probability": r["promotion_probability"],
                                  "champion_probability": r["champion_probability"],
                                  "avg_projected_position": r["avg_projected_position"],
                                  "elo_rating": r["elo_rating"]} for r in table}
        })

    history.sort(key=lambda s: s["matchday"])
    return history


# ── MAIN PIPELINE ────────────────────────────────────────────────────────────
def main():
    print("Fetching all match data from OpenLigaDB...")
    all_matches = fetch_all_matches()
    played = [m for m in all_matches if m["finished"]]
    print(f"  {len(played)} finished matches")

    # Sanity check: three full historical seasons alone are ~918 finished matches.
    # Far fewer means a fetch failed — stop loudly rather than publish wrong numbers.
    if len(played) < 900:
        raise RuntimeError(f"Only {len(played)} finished matches fetched, expected at least 900. "
                           f"Stopping rather than publishing incomplete/wrong data.")

    print("Running the model on current data...")
    state = model_snapshot(all_matches)
    print(f"  {len(state['teams'])} clubs, {len(state['fixtures'])} fixtures remaining, "
          f"{state['n_sim']} simulations")

    predicted_table = build_predicted_table(state)
    team_colors = fetch_team_colors(state["teams"])

    print("Building trajectories, benchmarks and fixture difficulty...")
    labels = [season_label(s) for s in SEASONS]
    trajectories = build_trajectories(all_matches, labels)
    benchmarks = build_promotion_benchmarks(all_matches, labels[:-1])
    fixture_difficulty = build_fixture_difficulty(state["fixtures"])

    print("Building report data (last season, analogues, recent results)...")
    last_season = build_last_season_comparison(all_matches, state)
    n_games = max((s["played"] for s in state["stats"].values()), default=0)
    analogues = build_historical_analogues(all_matches, labels[:-1], n_games)
    recent_results = build_recent_results(state)

    print("Updating history snapshots...")
    history = update_history(all_matches, load_history())
    os.makedirs(os.path.dirname(HISTORY_FILE), exist_ok=True)
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False)
    print(f"  {len(history)} snapshots stored")

    output = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "current_season": CURRENT_SEASON,
        "model_version": MODEL_VERSION,
        "n_simulations": state["n_sim"],
        "league_avg_goals": round(state["league_avg"], 3),
        "games_played_max": n_games,
        "games_remaining": len(state["fixtures"]),
        "predicted_table": predicted_table,
        "trajectories": trajectories,
        "promotion_benchmarks": benchmarks,
        "fixture_difficulty": fixture_difficulty,
        "last_season_comparison": last_season,
        "historical_analogues": analogues,
        "recent_results": recent_results,
        "history": history,
        "team_colors": team_colors,
        "methodology_note": "Fixture probabilities come from a Poisson attack/defense model "
                            "(last season plus this season so far, shrunk toward average for clubs "
                            "with few games). The rest of the season is simulated 10,000 times by "
                            "sampling scorelines, with official tie-breaks. Elo is shown for context "
                            "and is not an input to the predictions."
    }

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\nDone. Written to {OUTPUT_FILE}")
    for row in predicted_table[:5]:
        print(f"  {row['team']}: avg pos {row['avg_projected_position']}, "
              f"promotion {row['promotion_probability']*100:.1f}%")


if __name__ == "__main__":
    main()
