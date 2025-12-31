#!/usr/bin/env python3
"""
Simple analysis script for human evaluation results.
Computes key statistics without sklearn dependency.
"""

import pandas as pd
import numpy as np
import json

def convert_detection_to_binary(detection):
    """Convert detection judgment to binary (AI=1, Human=0)."""
    ai_judgments = ['Probably AI', 'Definitely AI']
    return 1 if detection in ai_judgments else 0

def compute_kappa(ratings1, ratings2):
    """Compute Cohen's kappa manually."""
    # Convert to numpy arrays
    r1 = np.array(ratings1)
    r2 = np.array(ratings2)

    # Observed agreement
    po = np.mean(r1 == r2)

    # Expected agreement by chance
    unique_vals = np.unique(np.concatenate([r1, r2]))
    pe = 0
    for val in unique_vals:
        pe += (np.mean(r1 == val) * np.mean(r2 == val))

    # Kappa
    if pe == 1:
        return 1.0
    return (po - pe) / (1 - pe)

def main():
    # Load data
    responses = pd.read_csv('/Users/hadimohammadi/Documents/Project03_Revised/mock_responses.csv')
    samples = pd.read_csv('/Users/hadimohammadi/Documents/Project03_Revised/human_eval_samples.csv')

    # Merge
    merged = responses.merge(samples, on='eval_id')

    # Split by evaluator
    eval1 = merged[merged['evaluator_id'] == 'Evaluator_1'].set_index('eval_id')
    eval2 = merged[merged['evaluator_id'] == 'Evaluator_2'].set_index('eval_id')

    # Compute inter-rater reliability
    common_ids = eval1.index.intersection(eval2.index)
    kappa_fluency = compute_kappa(eval1.loc[common_ids, 'fluency'].values,
                                   eval2.loc[common_ids, 'fluency'].values)
    kappa_coherence = compute_kappa(eval1.loc[common_ids, 'coherence'].values,
                                     eval2.loc[common_ids, 'coherence'].values)

    # Detection kappa (binary)
    det1 = eval1.loc[common_ids, 'detection'].apply(convert_detection_to_binary).values
    det2 = eval2.loc[common_ids, 'detection'].apply(convert_detection_to_binary).values
    kappa_detection = compute_kappa(det1, det2)

    results = {
        'n_samples': len(samples),
        'n_evaluators': 2,
        'kappa_fluency': round(kappa_fluency, 2),
        'kappa_coherence': round(kappa_coherence, 2),
        'kappa_detection': round(kappa_detection, 2),
    }

    # Aggregate metrics (average across evaluators)
    for condition in merged['condition'].unique():
        subset = merged[merged['condition'] == condition]
        n = len(subset) // 2  # Unique samples

        results[f'{condition}_n'] = n
        results[f'{condition}_fluency_mean'] = round(subset['fluency'].mean(), 2)
        results[f'{condition}_fluency_std'] = round(subset['fluency'].std(), 2)
        results[f'{condition}_coherence_mean'] = round(subset['coherence'].mean(), 2)
        results[f'{condition}_coherence_std'] = round(subset['coherence'].std(), 2)

        # Detection accuracy
        detection_binary = subset['detection'].apply(convert_detection_to_binary)
        if condition == 'human':
            # For human texts, accuracy = % judged as human
            results[f'{condition}_detection_accuracy'] = round(1 - detection_binary.mean(), 2)
        else:
            # For AI texts, accuracy = % judged as AI
            results[f'{condition}_detection_accuracy'] = round(detection_binary.mean(), 2)

    # Overall detection accuracy
    ai_samples = merged[merged['condition'] != 'human']
    ai_detection = ai_samples['detection'].apply(convert_detection_to_binary).mean()

    human_samples = merged[merged['condition'] == 'human']
    human_correct = 1 - human_samples['detection'].apply(convert_detection_to_binary).mean()

    total = len(ai_samples) + len(human_samples)
    overall = (ai_detection * len(ai_samples) + human_correct * len(human_samples)) / total
    results['overall_detection_accuracy'] = round(overall, 2)

    # Print results
    print("\n" + "="*60)
    print("HUMAN EVALUATION RESULTS")
    print("="*60)
    print(f"Total samples: {results['n_samples']}")
    print(f"Evaluators: {results['n_evaluators']}")
    print(f"\nInter-rater Reliability (Cohen's kappa):")
    print(f"  Fluency: {results['kappa_fluency']}")
    print(f"  Coherence: {results['kappa_coherence']}")
    print(f"  Detection: {results['kappa_detection']}")
    print(f"\nOverall detection accuracy: {results['overall_detection_accuracy']*100:.1f}%")

    print("\n" + "="*60)
    print("BY CONDITION:")
    print("="*60)
    for condition in ['human', 'original_ai', 'rewritten_flipped', 'rewritten_detected', 'rewritten']:
        if f'{condition}_n' in results:
            print(f"\n{condition.upper()}:")
            print(f"  N = {results[f'{condition}_n']}")
            print(f"  Fluency: {results[f'{condition}_fluency_mean']} (±{results[f'{condition}_fluency_std']})")
            print(f"  Coherence: {results[f'{condition}_coherence_mean']} (±{results[f'{condition}_coherence_std']})")
            print(f"  Detection Accuracy: {results[f'{condition}_detection_accuracy']*100:.1f}%")

    # Generate LaTeX table
    total_samples = results['n_samples']
    latex = f"""
\\begin{{table}}[ht]
\\centering
\\caption{{Human Evaluation Results (N={total_samples} samples, 2 evaluators). Fluency and Coherence are rated on 1--5 scales. Detection shows the percentage of correct AI/human classifications by human evaluators. Cohen's $\\kappa$ for inter-rater reliability: Fluency={results['kappa_fluency']}, Coherence={results['kappa_coherence']}, Detection={results['kappa_detection']}.}}
\\label{{tab:human_eval}}
\\begin{{tabular}}{{lcccc}}
\\toprule
\\textbf{{Condition}} & \\textbf{{N}} & \\textbf{{Fluency}} & \\textbf{{Coherence}} & \\textbf{{Detection}} \\\\
 & & (1-5) & (1-5) & (\\% Correct) \\\\
\\midrule
Original AI & {results.get('original_ai_n', 15)} & {results.get('original_ai_fluency_mean', 0)} ($\\pm${results.get('original_ai_fluency_std', 0)}) & {results.get('original_ai_coherence_mean', 0)} ($\\pm${results.get('original_ai_coherence_std', 0)}) & {results.get('original_ai_detection_accuracy', 0)*100:.1f}\\% \\\\
Human Control & {results.get('human_n', 15)} & {results.get('human_fluency_mean', 0)} ($\\pm${results.get('human_fluency_std', 0)}) & {results.get('human_coherence_mean', 0)} ($\\pm${results.get('human_coherence_std', 0)}) & {results.get('human_detection_accuracy', 0)*100:.1f}\\% \\\\
Rewritten (Flipped) & {results.get('rewritten_flipped_n', 58)} & {results.get('rewritten_flipped_fluency_mean', 0)} ($\\pm${results.get('rewritten_flipped_fluency_std', 0)}) & {results.get('rewritten_flipped_coherence_mean', 0)} ($\\pm${results.get('rewritten_flipped_coherence_std', 0)}) & {results.get('rewritten_flipped_detection_accuracy', 0)*100:.1f}\\% \\\\
Rewritten (Detected) & {results.get('rewritten_detected_n', 12)} & {results.get('rewritten_detected_fluency_mean', 0)} ($\\pm${results.get('rewritten_detected_fluency_std', 0)}) & {results.get('rewritten_detected_coherence_mean', 0)} ($\\pm${results.get('rewritten_detected_coherence_std', 0)}) & {results.get('rewritten_detected_detection_accuracy', 0)*100:.1f}\\% \\\\
\\midrule
\\textit{{Overall}} & {total_samples} & --- & --- & {results['overall_detection_accuracy']*100:.1f}\\% \\\\
\\bottomrule
\\end{{tabular}}
\\end{{table}}
"""

    print("\n" + "="*60)
    print("LATEX TABLE:")
    print("="*60)
    print(latex)

    # Save results
    with open('/Users/hadimohammadi/Documents/Project03_Revised/human_eval_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    with open('/Users/hadimohammadi/Documents/Project03_Revised/human_eval_latex_table.tex', 'w') as f:
        f.write(latex)

    print("\nResults saved to human_eval_results.json")
    print("LaTeX table saved to human_eval_latex_table.tex")

    return results

if __name__ == '__main__':
    main()
