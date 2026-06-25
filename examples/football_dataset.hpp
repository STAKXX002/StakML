#pragma once
#include <stakml/tensor.hpp>
#include <string>
#include <fstream>
#include <vector>
#include <sstream>
#include <unordered_map>
#include <cmath>
#include <stdexcept>

// ─────────────────────────────────────────────────────────────────────────────
// football_dataset.hpp — Football match data loader
//
// Domain-specific loader for the international football results CSV dataset.
// Lives in examples/ rather than include/stakml/ because it is application
// code, not general-purpose library code.
//
// Used by: world_cup.cpp, predict_match.cpp, group_stage.cpp, full_tournament.cpp
// ─────────────────────────────────────────────────────────────────────────────

namespace stakml {
namespace dataset {

struct TeamStats {
    float elo = 1500.0f;
    float ema_goals = 1.0f;

    void update_elo(float actual_score, float expected_score, float tournament_weight) {
        float k_factor = 20.0f + (tournament_weight * 40.0f);
        elo += k_factor * (actual_score - expected_score);
    }

    void update_goals(int goals_scored) {
        ema_goals = (0.2f * goals_scored) + (0.8f * ema_goals);
    }
};

struct FootballMatch {
    int home_team_id;
    int away_team_id;
    int outcome;

    float is_neutral;
    float tournament_weight;
    float home_elo;
    float away_elo;
    float home_ema_goals;
    float away_ema_goals;
};

struct Football {
    std::vector<FootballMatch> matches;
    std::unordered_map<std::string, int> team_to_id;
    std::vector<std::string> id_to_team;
    std::unordered_map<int, TeamStats> current_stats;

    int get_or_add_team(const std::string& name) {
        auto it = team_to_id.find(name);
        if (it != team_to_id.end()) return it->second;
        int new_id = id_to_team.size();
        team_to_id[name] = new_id;
        id_to_team.push_back(name);
        return new_id;
    }

    static Football load(const std::string& filepath) {
        Football dataset;
        std::ifstream file(filepath);
        if (!file.is_open())
            throw std::runtime_error("Could not open " + filepath);

        std::string line;
        std::getline(file, line); // skip header

        while (std::getline(file, line)) {
            std::stringstream ss(line);
            std::string date, home, away, h_score_str, a_score_str,
                        tournament, city, country, neutral_str;

            std::getline(ss, date,        ',');
            std::getline(ss, home,        ',');
            std::getline(ss, away,        ',');
            std::getline(ss, h_score_str, ',');
            std::getline(ss, a_score_str, ',');
            std::getline(ss, tournament,  ',');
            std::getline(ss, city,        ',');
            std::getline(ss, country,     ',');
            std::getline(ss, neutral_str, '\r');

            if (h_score_str.empty() || a_score_str.empty()) continue;

            int h_score = 0, a_score = 0;
            try {
                h_score = std::stoi(h_score_str);
                a_score = std::stoi(a_score_str);
            } catch (...) { continue; }

            int h_id = dataset.get_or_add_team(home);
            int a_id = dataset.get_or_add_team(away);

            FootballMatch m;
            m.home_team_id = h_id;
            m.away_team_id = a_id;
            m.outcome      = (h_score > a_score) ? 0 : (a_score > h_score) ? 1 : 2;
            m.is_neutral   = (neutral_str == "TRUE") ? 1.0f : 0.0f;

            if      (tournament == "Friendly")       m.tournament_weight = 0.0f;
            else if (tournament == "FIFA World Cup")  m.tournament_weight = 1.0f;
            else                                      m.tournament_weight = 0.5f;

            m.home_elo       = dataset.current_stats[h_id].elo;
            m.away_elo       = dataset.current_stats[a_id].elo;
            m.home_ema_goals = dataset.current_stats[h_id].ema_goals;
            m.away_ema_goals = dataset.current_stats[a_id].ema_goals;

            dataset.matches.push_back(m);

            float expected_h = 1.0f / (1.0f + std::pow(10.0f, (m.away_elo - m.home_elo) / 400.0f));
            float expected_a = 1.0f - expected_h;
            float score_h    = (m.outcome == 0) ? 1.0f : (m.outcome == 2 ? 0.5f : 0.0f);
            float score_a    = (m.outcome == 1) ? 1.0f : (m.outcome == 2 ? 0.5f : 0.0f);

            dataset.current_stats[h_id].update_elo(score_h, expected_h, m.tournament_weight);
            dataset.current_stats[a_id].update_elo(score_a, expected_a, m.tournament_weight);
            dataset.current_stats[h_id].update_goals(h_score);
            dataset.current_stats[a_id].update_goals(a_score);
        }
        return dataset;
    }
};

} // namespace dataset
} // namespace stakml