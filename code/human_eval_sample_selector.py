#!/usr/bin/env python3
"""
human_eval_sample_selector.py
Selects samples for human evaluation from storage_1_tb_new data.

Usage: python human_eval_sample_selector.py
"""

import pandas as pd
import random
import os

# Configuration
DATA_DIR = '/Users/hadimohammadi/Documents/Project03_Revised/storage_1_tb_new'

def load_all_data():
    """Load all strategy results and original data."""
    strategies = {}
    for i in range(1, 5):
        strategies[i] = pd.read_csv(f'{DATA_DIR}/strategy{i}_results.csv')
        strategies[i]['strategy'] = i
        strategies[i]['strategy_name'] = ['HSR', 'PSR', 'GPT', 'GPT+Genre'][i-1]

    # Load original data for human samples
    original_data = []
    for lang in ['en', 'nl']:
        for domain in ['news', 'reviews', 'twitter']:
            path = f'{DATA_DIR}/Data/dev_{lang}_{domain}.csv'
            if os.path.exists(path):
                df = pd.read_csv(path)
                df['lang'] = lang
                df['domain'] = domain
                original_data.append(df)

    return strategies, pd.concat(original_data, ignore_index=True)

def select_samples():
    """Select 100 samples for human evaluation (~10% coverage)."""
    strategies, original = load_all_data()
    selected = []

    random.seed(42)  # For reproducibility

    # Distribution for 100 samples:
    # - HSR flipped: 18
    # - PSR flipped: 14
    # - GPT flipped: 14
    # - GPT+Genre flipped: 12
    # - Detected (not flipped): 12
    # - Original AI: 15
    # - Human control: 15
    # Total: 100 samples

    # 1. HSR Flipped samples (18)
    hsr_flipped = strategies[1][(strategies[1]['true_label']==1) & (strategies[1]['pred_label']==0)]
    for _, row in hsr_flipped.sample(min(18, len(hsr_flipped))).iterrows():
        selected.append({
            'eval_id': f'TEXT_{len(selected)+1:03d}',
            'text': row['modified_text'],
            'original_text': row['original_text'],
            'replacements': row['replacements'],
            'category': 'hsr_flipped',
            'strategy': 'HSR',
            'condition': 'rewritten_flipped'
        })

    # 2. PSR Flipped samples (14)
    psr_flipped = strategies[2][(strategies[2]['true_label']==1) & (strategies[2]['pred_label']==0)]
    for _, row in psr_flipped.sample(min(14, len(psr_flipped))).iterrows():
        selected.append({
            'eval_id': f'TEXT_{len(selected)+1:03d}',
            'text': row['modified_text'],
            'original_text': row['original_text'],
            'replacements': row['replacements'],
            'category': 'psr_flipped',
            'strategy': 'PSR',
            'condition': 'rewritten_flipped'
        })

    # 3. GPT Flipped samples (14)
    gpt_flipped = strategies[3][(strategies[3]['true_label']==1) & (strategies[3]['pred_label']==0)]
    for _, row in gpt_flipped.sample(min(14, len(gpt_flipped))).iterrows():
        selected.append({
            'eval_id': f'TEXT_{len(selected)+1:03d}',
            'text': row['modified_text'],
            'original_text': row['original_text'],
            'replacements': row['replacements'],
            'category': 'gpt_flipped',
            'strategy': 'GPT',
            'condition': 'rewritten_flipped'
        })

    # 4. GPT+Genre Flipped samples (12)
    gptg_flipped = strategies[4][(strategies[4]['true_label']==1) & (strategies[4]['pred_label']==0)]
    for _, row in gptg_flipped.sample(min(12, len(gptg_flipped))).iterrows():
        selected.append({
            'eval_id': f'TEXT_{len(selected)+1:03d}',
            'text': row['modified_text'],
            'original_text': row['original_text'],
            'replacements': row['replacements'],
            'category': 'gptgenre_flipped',
            'strategy': 'GPT+Genre',
            'condition': 'rewritten_flipped'
        })

    # 5. Detected (not flipped) samples (12) - from all strategies
    for strat_id, count in [(1, 4), (2, 3), (3, 3), (4, 2)]:
        detected = strategies[strat_id][(strategies[strat_id]['true_label']==1) & (strategies[strat_id]['pred_label']==1)]
        for _, row in detected.sample(min(count, len(detected))).iterrows():
            selected.append({
                'eval_id': f'TEXT_{len(selected)+1:03d}',
                'text': row['modified_text'],
                'original_text': row['original_text'],
                'replacements': row['replacements'],
                'category': 'detected',
                'strategy': ['HSR', 'PSR', 'GPT', 'GPT+Genre'][strat_id-1],
                'condition': 'rewritten_detected'
            })

    # 6. Original AI (unmodified) samples (15)
    ai_original = original[original['label'] == 1].sample(15)
    for _, row in ai_original.iterrows():
        selected.append({
            'eval_id': f'TEXT_{len(selected)+1:03d}',
            'text': row['text'],
            'original_text': row['text'],
            'replacements': '{}',
            'category': 'original_ai',
            'strategy': 'None',
            'condition': 'original_ai'
        })

    # 7. Human-written samples (15)
    human = original[original['label'] == 0].sample(15)
    for _, row in human.iterrows():
        selected.append({
            'eval_id': f'TEXT_{len(selected)+1:03d}',
            'text': row['text'],
            'original_text': row['text'],
            'replacements': '{}',
            'category': 'human',
            'strategy': 'None',
            'condition': 'human'
        })

    # Shuffle for blind evaluation
    random.shuffle(selected)

    # Reassign eval_ids after shuffle (3 digits for 100 samples)
    for i, sample in enumerate(selected):
        sample['eval_id'] = f'TEXT_{i+1:03d}'

    return pd.DataFrame(selected)

if __name__ == '__main__':
    selected = select_samples()

    # Save full metadata (for analysis)
    output_dir = '/Users/hadimohammadi/Documents/Project03_Revised'
    selected.to_csv(f'{output_dir}/human_eval_samples.csv', index=False)
    print(f"Saved {len(selected)} samples to human_eval_samples.csv")

    # Save texts only (for Google Form)
    texts_only = selected[['eval_id', 'text']].copy()
    texts_only.to_csv(f'{output_dir}/human_eval_texts_only.csv', index=False)
    print(f"Saved texts to human_eval_texts_only.csv")

    # Print summary
    print("\nSample Distribution:")
    print(selected['category'].value_counts())
    print(f"\nTotal: {len(selected)} samples")

    # Print condition distribution
    print("\nCondition Distribution:")
    print(selected['condition'].value_counts())
