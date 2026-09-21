"""
2. Bundesliga Dashboard — Data & Predictions Pipeline
========================================================
This script does everything automatically, every time it runs:
  1. Pulls fresh match data for the last 4 seasons from OpenLigaDB (free, no key needed)
  2. Builds Elo ratings for every club from real match history
  3. Fits attack/defense strength for every current club (with shrinkage
     for teams with few games played this season)
  4. Runs a 10,000-simulation Monte Carlo projection of the rest of the season
  5. Writes everything the dashboard needs into data/dashboard_data.json

This is designed to be run by GitHub Actions on a schedule — see the
workflow file (.github/workflows/update_data.yml) for how it's triggered.
No manual steps are needed once that's set up.
"""

import requests
import json
import time
import numpy as np
from scipy.stats import poisson
from datetime import datetime, timezone

# ── CONFIG ──────────────────────────────────────────────────────────────────
SEASONS = [2023, 2024, 2025, 2026]          # 2023 = 2023/24 season, etc.
LEAGUE = "bl2"                               # 2. Bundesliga on OpenLigaDB
CURRENT_SEASON = "2026/27"
K_FACTOR = 20                                # Elo sensitivity to each result
HOME_ADVANTAGE_ELO = 60                      # Elo points of home advantage
SEASON_REGRESSION = 0.75                     # ratings pulled 25% toward average each new season
SHRINKAGE_WEIGHT = 10                        # how much a "league average" prior counts, in matches
HOME_GOAL_BOOST = 1.12                       # home teams score ~12% more on average
N_SIMULATIONS = 10000

np.random.seed(None)  # genuinely random each run, not fixed like our test


def season_label(year):
    return f"{year}/{str(year + 1)[-2:]}"


# ── STEP 1: PULL ALL MATCH DATA ──────────────────────────────────────────────
def fetch_all_matches():
    """Pulls each season's ENTIRE match list in a single request, rather than
    looping through 34 individual matchday requests. This matters because
    OpenLigaDB enforces a shared rate limit (1000 requests/hour per IP), and
    GitHub Actions runners share IP pools with many other unrelated users'
    workflows — 136 requests per run was hitting that shared ceiling. 4
    requests total (one per season) is dramatically safer."""
    all_matches = []

    for season in SEASONS:
        label = season_label(season)

        url = f"https://api.openligadb.de/getmatchdata/{LEAGUE}/{season}"
        matches = None

        for attempt in range(3):  # a little more resilient since this one request matters a lot
            try:
                resp = requests.get(url, timeout=20)
                if resp.status_code == 200:
                    matches = resp.json()
                    break
                else:
                    print(f"  WARNING: season {label} returned HTTP {resp.status_code} (attempt {attempt+1})")
            except requests.RequestException as e:
                print(f"  WARNING: season {label} request failed: {e} (attempt {attempt+1})")
            time.sleep(2)  # a real pause before retrying, not just a courtesy delay

        time.sleep(1)  # be polite between seasons too

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

            home_goals = final.get("pointsTeam1") if (is_finished and final) else None
            away_goals = final.get("pointsTeam2") if (is_finished and final) else None

            date_str = m.get("matchDateTime", "")[:10] if m.get("matchDateTime") else ""

            all_matches.append({
                "season": label,
                "matchday": matchday,
                "date": date_str,
                "home": home,
                "away": away,
                "home_goals": home_goals,
                "away_goals": away_goals,
                "finished": is_finished
            })

        finished_count = sum(1 for m in matches if m.get("matchIsFinished"))
        print(f"  {label}: {len(matches)} total matches fetched, {finished_count} finished")

    return all_matches


# ── STEP 2: ELO RATINGS ───────────────────────────────────────────────────────
def expected_score(ra, rb):
    return 1.0 / (1.0 + 10 ** ((rb - ra) / 400))


def build_elo_ratings(played_matches):
    """Runs chronologically through every finished match, updating ratings.
    New teams start at 1500. Ratings regress toward the mean between seasons
    to account for squad turnover."""
    ratings = {}
    current_season = None

    # sort chronologically
    played_sorted = sorted(played_matches, key=lambda m: (m["season"], m["date"]))

    for m in played_sorted:
        season, home, away = m["season"], m["home"], m["away"]
        hg, ag = m["home_goals"], m["away_goals"]
        if hg is None or ag is None:
            continue

        if home not in ratings:
            ratings[home] = 1500.0
        if away not in ratings:
            ratings[away] = 1500.0

        if season != current_season:
            if current_season is not None:
                for t in ratings:
                    ratings[t] = 1500 + (ratings[t] - 1500) * SEASON_REGRESSION
            current_season = season

        ra, rb = ratings[home], ratings[away]
        exp_home = expected_score(ra + HOME_ADVANTAGE_ELO, rb)

        if hg > ag:
            result_home = 1.0
        elif hg == ag:
            result_home = 0.5
        else:
            result_home = 0.0

        ratings[home] = ra + K_FACTOR * (result_home - exp_home)
        ratings[away] = rb + K_FACTOR * ((1 - result_home) - (1 - exp_home))

    return ratings


# ── STEP 3: ATTACK / DEFENSE, WITH SHRINKAGE ─────────────────────────────────
def build_attack_defense(played_matches, current_teams):
    """Uses the last two seasons of data (last season + this season so far)
    to estimate each current team's attacking and defensive strength,
    shrinking teams with few games toward the league average."""
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
    league_avg = np.mean(all_goals) if all_goals else 1.5

    attack, defense = {}, {}
    for t in current_teams:
        n = len(goals_scored[t])
        if n == 0:
            attack[t], defense[t] = 1.0, 1.0
            continue
        raw_attack = np.mean(goals_scored[t]) / league_avg
        raw_defense = np.mean(goals_conceded[t]) / league_avg
        attack[t] = (n * raw_attack + SHRINKAGE_WEIGHT * 1.0) / (n + SHRINKAGE_WEIGHT)
        defense[t] = (n * raw_defense + SHRINKAGE_WEIGHT * 1.0) / (n + SHRINKAGE_WEIGHT)

    return attack, defense, league_avg


# ── STEP 3b: TEAM COLORS (for the dynamic dashboard theme) ─────────────────
DEFAULT_COLORS = {"primary": "#1a1a2e", "secondary": "#e0e0e0", "text": "#ffffff"}


def fetch_team_colors(team_names):
    """Looks up each club's official brand colors from Sofascore.
    If a lookup fails for any team, that team just gets a safe neutral
    fallback color rather than breaking the whole pipeline."""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                      "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "application/json"
    }

    colors = {}
    failed = []
    debug_printed = False  # only print full diagnostic detail for the first team, to avoid a huge log

    for name in team_names:
        try:
            url = f"https://api.sofascore.com/api/v1/search/all?q={requests.utils.quote(name)}"
            resp = requests.get(url, headers=headers, timeout=10)

            if not debug_printed:
                print(f"  DEBUG (first lookup only) — team: {name}")
                print(f"  DEBUG — status code: {resp.status_code}")
                print(f"  DEBUG — response body (first 500 chars): {resp.text[:500]}")
                debug_printed = True

            if resp.status_code != 200:
                failed.append(name)
                colors[name] = DEFAULT_COLORS
                time.sleep(0.3)
                continue

            data = resp.json()
            team_colors = None

            for result in data.get("results", []):
                entity = result.get("entity", {})
                if entity.get("sport", {}).get("slug") == "football" and "teamColors" in entity:
                    team_colors = entity["teamColors"]
                    break

            colors[name] = team_colors if team_colors else DEFAULT_COLORS
            if not team_colors:
                failed.append(name)

        except (requests.RequestException, ValueError, KeyError) as e:
            colors[name] = DEFAULT_COLORS
            failed.append(name)
            if not debug_printed:
                print(f"  DEBUG (first lookup only) — team: {name}, exception: {repr(e)}")
                debug_printed = True

        time.sleep(0.3)  # be polite, and avoid the same rate-limit issue as before

    if failed:
        print(f"  NOTE: used fallback color for {len(failed)} team(s) — lookup didn't "
              f"return a match: {failed}")

    return colors


def expected_goals(home, away, attack, defense, league_avg):
    eg_home = league_avg * attack[home] * defense[away] * HOME_GOAL_BOOST
    eg_away = league_avg * attack[away] * defense[home]
    return eg_home, eg_away


def match_outcome_probs(eg_home, eg_away, max_goals=8):
    home_probs = [poisson.pmf(i, eg_home) for i in range(max_goals + 1)]
    away_probs = [poisson.pmf(i, eg_away) for i in range(max_goals + 1)]
    hw = dr = aw = 0.0
    for i in range(max_goals + 1):
        for j in range(max_goals + 1):
            p = home_probs[i] * away_probs[j]
            if i > j: hw += p
            elif i == j: dr += p
            else: aw += p
    return hw, dr, aw


def run_monte_carlo(current_points, current_gd, remaining_fixtures, fixture_probs, teams, n_sim=N_SIMULATIONS):
    position_counts = {t: {p: 0 for p in range(1, len(teams) + 1)} for t in teams}

    for _ in range(n_sim):
        sim_pts = dict(current_points)
        sim_gd = dict(current_gd)

        for home, away in remaining_fixtures:
            hw, dr, aw = fixture_probs[(home, away)]
            r = np.random.random()
            if r < hw:
                sim_pts[home] += 3
            elif r < hw + dr:
                sim_pts[home] += 1
                sim_pts[away] += 1
            else:
                sim_pts[away] += 3

        ranked = sorted(teams, key=lambda t: (sim_pts[t], sim_gd[t], np.random.random()), reverse=True)
        for pos, t in enumerate(ranked, start=1):
            position_counts[t][pos] += 1

    return position_counts


# ── STEP 5: TRAJECTORY DATA (for the chart) ──────────────────────────────────
def build_trajectories(all_matches, seasons_wanted):
    """Cumulative points by matchday, for every team, for the requested seasons."""
    trajectories = {}
    for label in seasons_wanted:
        season_matches = [m for m in all_matches if m["season"] == label and m["finished"]]
        season_matches.sort(key=lambda m: m["matchday"])

        team_points = {}
        team_trajectory = {}

        for m in season_matches:
            h, a, hg, ag = m["home"], m["away"], m["home_goals"], m["away_goals"]
            for t in (h, a):
                if t not in team_points:
                    team_points[t] = 0
                    team_trajectory[t] = []

            if hg > ag:
                team_points[h] += 3
            elif hg < ag:
                team_points[a] += 3
            else:
                team_points[h] += 1
                team_points[a] += 1

            team_trajectory[h].append({"matchday": m["matchday"], "points": team_points[h]})
            team_trajectory[a].append({"matchday": m["matchday"], "points": team_points[a]})

        trajectories[label] = team_trajectory

    return trajectories


# ── MAIN PIPELINE ────────────────────────────────────────────────────────────
def main():
    print("Fetching all match data from OpenLigaDB...")
    all_matches = fetch_all_matches()
    played = [m for m in all_matches if m["finished"]]
    upcoming = [m for m in all_matches if not m["finished"] and m["season"] == CURRENT_SEASON]
    print(f"  {len(played)} finished matches, {len(upcoming)} upcoming fixtures this season")

    # Sanity check: 3 full historical seasons alone should be ~918 finished matches
    # (306 each). If we're well below that, something failed silently upstream —
    # better to stop loudly here than publish wrong predictions.
    EXPECTED_MINIMUM = 900
    if len(played) < EXPECTED_MINIMUM:
        raise RuntimeError(
            f"Only {len(played)} finished matches fetched, expected at least "
            f"{EXPECTED_MINIMUM}. Likely a fetch failure (see warnings above) — "
            f"stopping rather than publishing incomplete/wrong data."
        )

    print("Building Elo ratings from full history...")
    elo_ratings = build_elo_ratings(played)

    current_teams = sorted(set(m["home"] for m in upcoming) | set(m["away"] for m in upcoming))
    print(f"  {len(current_teams)} teams in the current season")

    print("Fitting attack/defense strength (with shrinkage)...")
    attack, defense, league_avg = build_attack_defense(played, current_teams)

    print("Fetching official team colors...")
    team_colors = fetch_team_colors(current_teams)

    print("Computing current league table...")
    current_points = {t: 0 for t in current_teams}
    current_gd = {t: 0 for t in current_teams}
    played_this_season = [m for m in played if m["season"] == CURRENT_SEASON]

    for m in played_this_season:
        h, a, hg, ag = m["home"], m["away"], m["home_goals"], m["away_goals"]
        if hg > ag: current_points[h] += 3
        elif hg < ag: current_points[a] += 3
        else: current_points[h] += 1; current_points[a] += 1
        current_gd[h] += (hg - ag)
        current_gd[a] += (ag - hg)

    print("Computing fixture probabilities for every remaining match...")
    remaining_fixtures = [(m["home"], m["away"]) for m in upcoming]
    fixture_probs = {}
    for h, a in remaining_fixtures:
        eg_h, eg_a = expected_goals(h, a, attack, defense, league_avg)
        fixture_probs[(h, a)] = match_outcome_probs(eg_h, eg_a)

    print(f"Running {N_SIMULATIONS} season simulations...")
    position_counts = run_monte_carlo(current_points, current_gd, remaining_fixtures, fixture_probs, current_teams)

    print("Building trajectory data for last 3 seasons + current...")
    all_season_labels = [season_label(s) for s in SEASONS]
    trajectories = build_trajectories(all_matches, all_season_labels)

    # ── ASSEMBLE OUTPUT ──────────────────────────────────────────────────────
    predicted_table = []
    for t in current_teams:
        counts = position_counts[t]
        avg_pos = sum(pos * cnt for pos, cnt in counts.items()) / N_SIMULATIONS
        p_top2 = (counts[1] + counts[2]) / N_SIMULATIONS
        p_first = counts[1] / N_SIMULATIONS
        predicted_table.append({
            "team": t,
            "current_points": current_points[t],
            "current_gd": current_gd[t],
            "avg_projected_position": round(avg_pos, 2),
            "promotion_probability": round(p_top2, 4),
            "champion_probability": round(p_first, 4),
            "elo_rating": round(elo_ratings.get(t, 1500), 1),
            "attack_rating": round(attack[t], 3),
            "defense_rating": round(defense[t], 3)
        })

    predicted_table.sort(key=lambda x: x["avg_projected_position"])

    output = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "current_season": CURRENT_SEASON,
        "n_simulations": N_SIMULATIONS,
        "league_avg_goals": round(league_avg, 3),
        "predicted_table": predicted_table,
        "trajectories": trajectories,
        "team_colors": team_colors,
        "methodology_note": "Predictions from Elo ratings + Poisson attack/defense model "
                             "(shrinkage-adjusted for teams with limited current-season data) "
                             "+ Monte Carlo simulation of all remaining fixtures."
    }

    with open("data/dashboard_data.json", "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print("\nDone. Written to data/dashboard_data.json")
    print("\nTop 5 projected:")
    for row in predicted_table[:5]:
        print(f"  {row['team']}: avg pos {row['avg_projected_position']}, "
              f"promotion {row['promotion_probability']*100:.1f}%")


if __name__ == "__main__":
    main()
