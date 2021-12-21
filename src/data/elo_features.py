from typing import Dict, List, Tuple, Union

import numpy as np
import pandas as pd
from tqdm import tqdm


def get_new_theta(
    is_good_answer: int,
    beta: float,
    left_asymptote: float,
    theta: float,
    nb_previous_answers: float,
) -> float:
    return theta + learning_rate_theta(nb_previous_answers) * (
        is_good_answer - probability_of_good_answer(theta, beta, left_asymptote)
    )


def get_new_beta(
    is_good_answer: int,
    beta: float,
    left_asymptote: float,
    theta: float,
    nb_previous_answers: float,
) -> float:
    return beta - learning_rate_beta(nb_previous_answers) * (
        is_good_answer - probability_of_good_answer(theta, beta, left_asymptote)
    )


def learning_rate_theta(nb_answers: float) -> float:
    return max(0.3 / (1 + 0.01 * nb_answers), 0.04)


def learning_rate_beta(nb_answers: float) -> float:
    return 1 / (1 + 0.05 * nb_answers)


def probability_of_good_answer(
    theta: float, beta: float, left_asymptote: float
) -> float:
    return left_asymptote + (1 - left_asymptote) * sigmoid(theta - beta)


def sigmoid(x: Union[int, float]) -> float:
    return 1 / (1 + np.exp(-x))


def estimate_parameters(
    df: pd.DataFrame, granularity_feature_name: str = "contents_rn"
) -> Tuple[Dict[str, Union[int, float]]]:
    content_parameters = {
        granularity_feature_value: {"beta": 0, "nb_answers": 0}
        for granularity_feature_value in np.unique(df[granularity_feature_name])
    }
    person_parameters = {
        person_id: {"theta": 0, "nb_answers": 0}
        for person_id in np.unique(df.person_rn)
    }

    print("Parameter estimation is starting...")

    for person_id, content_id, left_asymptote, answered_correctly in tqdm(
        zip(
            df.person_rn.values,
            df[granularity_feature_name].values,
            df.left_asymptote.values,
            df.target.values,
        )
    ):
        theta = person_parameters[person_id]["theta"]
        beta = content_parameters[content_id]["beta"]

        content_parameters[content_id]["beta"] = get_new_beta(
            answered_correctly,
            beta,
            left_asymptote,
            theta,
            content_parameters[content_id]["nb_answers"],
        )
        person_parameters[person_id]["theta"] = get_new_theta(
            answered_correctly,
            beta,
            left_asymptote,
            theta,
            person_parameters[person_id]["nb_answers"],
        )

        content_parameters[content_id]["nb_answers"] += 1
        person_parameters[person_id]["nb_answers"] += 1

    print(f"Theta & beta estimations on {granularity_feature_name} are completed.")
    return person_parameters, content_parameters


def update_parameters(
    df: pd.DataFrame,
    person_parameters: Dict[str, Union[int, float]],
    content_parameters: Dict[str, Union[int, float]],
    granularity_feature_name="contents_rn",
) -> Tuple[Dict[str, Union[int, float]]]:
    for person_id, item_id, left_asymptote, answered_correctly in tqdm(
        zip(
            df.person_rn.values,
            df[granularity_feature_name].values,
            df.left_asymptote.values,
            df.target.values,
        )
    ):
        if person_id not in person_parameters:
            person_parameters[person_id] = {"theta": 0, "nb_answers": 0}
        if item_id not in content_parameters:
            content_parameters[item_id] = {"beta": 0, "nb_answers": 0}

        theta = person_parameters[person_id]["theta"]
        beta = content_parameters[item_id]["beta"]

        person_parameters[person_id]["theta"] = get_new_theta(
            answered_correctly,
            beta,
            left_asymptote,
            theta,
            person_parameters[person_id]["nb_answers"],
        )
        content_parameters[item_id]["beta"] = get_new_beta(
            answered_correctly,
            beta,
            left_asymptote,
            theta,
            content_parameters[item_id]["nb_answers"],
        )

        person_parameters[person_id]["nb_answers"] += 1
        content_parameters[item_id]["nb_answers"] += 1

    return person_parameters, content_parameters


def estimate_probas(
    df: pd.DataFrame,
    person_parameters: Dict[str, Union[int, float]],
    content_parameters: Dict[str, Union[int, float]],
    granularity_feature_name: str = "contents_rn",
) -> List[float]:
    probability_of_success_list = []

    for student_id, item_id, left_asymptote in tqdm(
        zip(
            df.person_rn.values,
            df[granularity_feature_name].values,
            df.left_asymptote.values,
        )
    ):
        theta = (
            person_parameters[student_id]["theta"]
            if student_id in person_parameters
            else 0
        )
        beta = (
            content_parameters[item_id]["beta"] if item_id in content_parameters else 0
        )

        probability_of_success_list.append(
            probability_of_good_answer(theta, beta, left_asymptote)
        )

    return probability_of_success_list
