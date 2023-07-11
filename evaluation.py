import pandas as pd
import numpy as np
import argparse
from importlib import import_module
import json
import random
import sys
from time import time
import json

N_DISPLAY = 3
MISSING_PENALTY = 5

with open('data/tax.json', 'r') as f:
    tax = json.load(f)

with open('data/rev_tax.json', 'r') as f:
    rev_tax = json.load(f)

def main():
    parser = argparse.ArgumentParser(description='Meal autocomplete evaluation utility')
    parser.add_argument('training_set', type=str, metavar='TRAIN', help='training set file path')
    parser.add_argument('test_set', type=str, metavar='TEST', help='test set file path')
    parser.add_argument('autocomplete_module', type=str, metavar='AUTOCOMPLETE_MODULE', help='autocomplete module path')

    args = parser.parse_args()

    print('Preparing data')
    train_df = pd.read_csv(args.training_set)
    test_df = pd.read_csv(args.test_set)

    train_df['food_items'] = train_df['food_items'].apply(json.loads)
    test_df['food_items'] = test_df['food_items'].apply(json.loads)

    ac_module = import_module(args.autocomplete_module.replace('.py', ''))
    autocompleter = ac_module.Autocomplete()

    print('Calling Autocomplete().fit')
    t1 = time()
    autocompleter.fit(train_df)
    t2 = time()
    print('Done in %.2fs' % (t2 - t1))

    print('Evaluating on test set (%d records):' % (test_df.shape[0]))
    evaluate(autocompleter, test_df)


def evaluate(autocompleter, test_df):
    ac_clicks = []
    clean_clicks = []
    t = []

    done = 0

    for _, row in test_df.iterrows():
        if done > 0:
            progress_bar(done / test_df.shape[0])
        meal_ac_clicks, durations = \
            evaluate_meal(autocompleter, row['food_items'])
        ac_clicks.append(meal_ac_clicks)
        t += durations
        clean_clicks.append(MISSING_PENALTY * len(row['food_items']))
        done += 1

    progress_bar(1.)
    print('\nDone!\n')

    mean_ac_clicks = np.mean(ac_clicks)
    mean_clean_clicks = np.mean(clean_clicks)

    print('Mean clicks w/o autocomplete:\t%.2f' % mean_clean_clicks)
    print('Mean clicks w/ autocomplete:\t%.2f' % mean_ac_clicks)
    print('Improvement factor: %d%%' % calc_improvement(mean_ac_clicks, mean_clean_clicks))
    print('Mean execution time: %.2fms' % (1000 * np.mean(t)))

def calc_improvement(mean_ac_clicks, mean_clean_clicks):
    ratio = mean_clean_clicks / mean_ac_clicks
    coef = 1
    if ratio < 1:
        ratio = 1 / ratio
        coef = -1

    return coef * 100 * (ratio - 1)


def evaluate_meal(autocompleter, food_items):
    to_log = set(food_items)
    logged = set()
    focus_node = None
    total_clicks = 0
    tried_focus = set()
    t = []
    while len(to_log):
        action_taken = False
        t1 = time()
        options = autocompleter.autocomplete(logged, focus_node)[:N_DISPLAY]
        t2 = time()
        t.append(t2 - t1)
        bulls_eye = to_log.intersection(set(options))
        if bulls_eye:
            logged.add(list(bulls_eye)[0])
            to_log.remove(list(bulls_eye)[0])
            total_clicks += 1
            focus_node = None
            action_taken = True
            continue

        for item_to_log in to_log:
            focus_candidates = list(set(rev_tax[item_to_log]).intersection(set(options)))
            if focus_candidates and focus_candidates[0] not in tried_focus:
                focus_node = focus_candidates[0]
                tried_focus.add(focus_node)
                total_clicks += 1
                action_taken = True
                break

        if not action_taken:
            logging_item = random.choice(list(to_log))
            logged.add(logging_item)
            to_log.remove(logging_item)
            focus_node = None
            total_clicks += MISSING_PENALTY

    return total_clicks, t


def progress_bar(fraction):
    WIDTH = 30
    sys.stdout.write('\r') # Back to beginning of line
    full = int(fraction * WIDTH)
    sys.stdout.write(('[%s] %d%%' % (('#' * full).ljust(WIDTH, ' '), fraction * 100)).ljust(80, ' '))
    sys.stdout.flush()

if __name__ == '__main__':
    main()
