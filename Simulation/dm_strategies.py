import numpy as np
import json
import scipy

################################
# CONSTS
################################

REVIEWS = 0
BOT_ACTION = 1
USER_DECISION = 2


################################

def correct_action(information):
    if information["hotel_value"] >= 8:
        return 1
    else:
        return 0


def random_action(information):
    return np.random.randint(2)


def user_rational_action(information):
    if information["bot_message"] >= 8:
        return 1
    else:
        return 0


def user_picky(information):
    if information["bot_message"] >= 9:
        return 1
    else:
        return 0


def user_sloppy(information):
    if information["bot_message"] >= 7:
        return 1
    else:
        return 0


def user_short_t4t(information):
    if len(information["previous_rounds"]) == 0 \
            or (information["previous_rounds"][-1][BOT_ACTION] >= 8 and
                information["previous_rounds"][-1][REVIEWS].mean() >= 8) \
            or (information["previous_rounds"][-1][BOT_ACTION] < 8 and
                information["previous_rounds"][-1][REVIEWS].mean() < 8):  # cooperation
        if information["bot_message"] >= 8:  # good hotel
            return 1
        else:
            return 0
    else:
        return 0


def user_picky_short_t4t(information):
    if information["bot_message"] >= 9 or ((information["bot_message"] >= 8) and (
            len(information["previous_rounds"]) == 0 or (
            information["previous_rounds"][-1][REVIEWS].mean() >= 8))):  # good hotel
        return 1
    else:
        return 0


def user_hard_t4t(information):
    if len(information["previous_rounds"]) == 0 \
            or np.min(np.array([((r[BOT_ACTION] >= 8 and r[REVIEWS].mean() >= 8)
                                 or (r[BOT_ACTION] <= 8 and r[REVIEWS].mean() < 8)) for r in
                                information["previous_rounds"]])) == 1:  # cooperation
        if information["bot_message"] >= 8:  # good hotel
            return 1
        else:
            return 0
    else:
        return 0


def history_and_review_quality(history_window, quality_threshold):
    def func(information):
        if len(information["previous_rounds"]) == 0 \
                or history_window == 0 \
                or np.min(np.array([((r[BOT_ACTION] >= 8 and r[REVIEWS].mean() >= 8)
                                     or (r[BOT_ACTION] <= 8 and r[REVIEWS].mean() < 8)) for r in
                                    information["previous_rounds"][
                                    -history_window:]])) == 1:  # cooperation from *result's* perspective
            if information["bot_message"] >= quality_threshold:  # good hotel from user's perspective
                return 1
            else:
                return 0
        else:
            return 0

    return func


def topic_based(positive_topics, negative_topics, quality_threshold):
    def func(information):
        review_personal_score = information["bot_message"]
        for rank, topic in enumerate(positive_topics):
            review_personal_score += int(information["review_features"].loc[topic]) * 2 / (rank + 1)
        for rank, topic in enumerate(negative_topics):
            review_personal_score -= int(information["review_features"].loc[topic]) * 2 / (rank + 1)
        if review_personal_score >= quality_threshold:  # good hotel from user's perspective
            return 1
        else:
            return 0

    return func


def LLM_based(is_stochastic):
    with open(f"data/baseline_proba2go.txt", 'r') as file:
        proba2go = json.load(file)
        proba2go = {int(k): v for k, v in proba2go.items()}

    if is_stochastic:
        def func(information):
            review_llm_score = proba2go[information["review_id"]]
            return int(np.random.rand() <= review_llm_score)

        return func
    else:
        def func(information):
            review_llm_score = proba2go[information["review_id"]]
            return int(review_llm_score >= 0.5)

        return func


def consensus_decision(information, history_window, quality_threshold, positive_topics, negative_topics):
    decisions = [
        random_action(information),
        user_rational_action(information),
        user_short_t4t(information),
        user_picky_short_t4t(information),
        user_hard_t4t(information),
        history_and_review_quality(history_window, quality_threshold),
        topic_based(positive_topics, negative_topics, quality_threshold),
        LLM_based(True),
        LLM_based(False)]
    # If the majority of the decisions are 1, return 1, otherwise, return 0.
    if sum(decisions) >= len(decisions) / 2:
        return 1
    else:
        return 0


def based_confidence_intervals(information):
    confidence_level = 0.95
    rounds = information["previous_rounds"]
    if not rounds:
        return user_rational_action(information)
    # Let's see if the bot is trustful
    n = len(rounds)
    grade = 0
    for r in rounds:
        mean = r[REVIEWS].mean()
        std_dev = r[REVIEWS].std()
        t_value = scipy.stats.t.ppf(1 - (1 - confidence_level) / 2, n - 1)
        margin_of_error = t_value * std_dev / np.sqrt(n)
        lower_bound = mean - margin_of_error
        if lower_bound >= 8 and r[BOT_ACTION] == 1:
            grade += 1
    if grade >= 0.75 * n:  # In 3/4 rounds the bot helped statistically the DM
        return 1
    else:
        return 0


def based_t_tests(information):
    rounds = information["previous_rounds"]
    if not rounds:
        return user_rational_action(information)
    alpha = 0.05
    # Let's see if the bot is trustful
    n = len(rounds)
    grade = 0
    for r in rounds:
        t_stat, p_value = scipy.stats.ttest_1samp(r[REVIEWS], popmean=8, alternative='greater')
        if p_value < alpha and r[BOT_ACTION] == 1:
            grade += 1
    if grade >= 0.75 * n:  # In 3/4 rounds the bot helped statistically the DM
        return 1
    else:
        return 0


def based_bootstraps(information):
    rounds = information["previous_rounds"]
    if not rounds:
        return user_rational_action(information)
    alpha = 0.05
    # Let's see if the bot is trustful
    n = len(rounds)
    grade = 0
    for r in rounds:
        t_stat, p_value = scipy.stats.ttest_1samp(r[REVIEWS], popmean=8, alternative='greater')
        if p_value < alpha and r[BOT_ACTION] == 1:
            grade += 1
    if grade >= 0.75 * n:  # In 3/4 rounds the bot helped statistically the DM
        return 1
    else:
        return 0


def based_mean_scores(information):
    rounds = information["previous_rounds"]
    if not rounds:
        return user_rational_action(information)
    # Let's see if the bot is trustful
    n = len(rounds)
    grade = 0
    for r in rounds:
        round_mean = r[REVIEWS].mean()
        if round_mean >= 8 and r[BOT_ACTION] == 1:
            grade += 1
    if grade >= 0.75 * n:  # In 3/4 rounds the bot helped statistically the DM
        return 1
    else:
        return 0


def consensus_statistics_decision(information):
    decisions = [
        based_confidence_intervals(information),
        based_t_tests(information),
        based_bootstraps(information),
        based_mean_scores(information)
    ]
    # If the majority of the decisions are 1, return 1, otherwise, return 0.
    if sum(decisions) >= len(decisions) / 2:
        return 1
    else:
        return 0


def explore_and_exploit(information):
    rounds = information["previous_rounds"]
    n = len(rounds)
    if not rounds:
        return user_rational_action(information)

    def is_positive_outcome(decision, reviews):
        avg_review = reviews.mean()
        return (decision == 1 and avg_review >= 8) or (decision == 0 and avg_review < 8)

    positive_outcomes = sum(is_positive_outcome(r[USER_DECISION], r[REVIEWS]) for r in rounds)
    negative_outcomes = len(rounds) - positive_outcomes
    explore_rate = 0.2
    positive_influence = 0.1 if n < 5 else 0.05  # reduce positive influence if there are many rounds
    negative_influence = 0.1 if n < 5 else 0.15  # increase negative influence as more rounds are played
    explore_rate = explore_rate - (positive_outcomes * positive_influence) + (negative_outcomes * negative_influence)
    explore_rate = np.clip(explore_rate, 0.05, 0.95)
    if np.random.rand() >= explore_rate:
        return 1
    else:
        return 0


def based_ucb_decision(information):
    c = 2
    rounds = information["previous_rounds"]
    if not rounds:
        return user_rational_action(information)

    # count the number of times each decision (1: go to hotel, 0: don't go) has been made
    n = len(rounds)
    num_go = sum(1 for r in rounds if r[USER_DECISION] == 1)
    num_no_go = len(rounds) - num_go
    avg_go_reviews = np.mean([r[REVIEWS].mean() for r in rounds if r[USER_DECISION] == 1]) if num_go > 0 else 0
    avg_no_go_reviews = np.mean([r[REVIEWS].mean() for r in rounds if r[USER_DECISION] == 0]) if num_no_go > 0 else 0

    # UCB values
    def ucb_value(avg_reviews, num_decisions, n, c):
        if num_decisions == 0:
            return float('inf')  # Encourage trying untested options
        return avg_reviews + c * np.sqrt(np.log(n) / num_decisions)

    # UCB for going to the hotel and not going
    ucb_go = ucb_value(avg_go_reviews, num_go, n, c)
    ucb_no_go = ucb_value(avg_no_go_reviews, num_no_go, n, c)
    # Exploit the decision with the higher UCB value
    if ucb_go > ucb_no_go:
        return 1
    else:
        return 0


def llm_bot_diff(information):
    with open(f"data/baseline_proba2go.txt", 'r') as file:
        proba2go = json.load(file)
        proba2go = {int(k): v for k, v in proba2go.items()}
    review_llm_score = proba2go[information["review_id"]]
    review_bot_score = information["bot_message"]
    # Let's see if the bot is trustful
    difference = abs(review_llm_score - review_bot_score)
    if difference <= 2:
        if review_bot_score >= 8:
            return 1
        else:
            return 0
    else:
        if review_llm_score >= 8:
            return 1
        else:
            return 0
