import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_training_results(original_results, transformed_results, metric_name="perplexity", 
                       title="Model Performance", ylabel=None, save_path=None, connect_initial=True):
    """Plot metrics vs. training steps for both original and transformed models."""
    plt.figure(figsize=(12, 6))
    
    if metric_name == "perplexity":
        metric_key = "perplexity_history"
        if ylabel is None:
            ylabel = "Perplexity (lower is better)"
    else:  # CE loss
        metric_key = "ce_loss_history"
        if ylabel is None:
            ylabel = "Cross-Entropy Loss (lower is better)"
    
    # Check if we have any data to plot
    orig_has_data = len(original_results.get("step_history", [])) > 0 and len(original_results.get(metric_key, [])) > 0
    trans_has_data = len(transformed_results.get("step_history", [])) > 0 and len(transformed_results.get(metric_key, [])) > 0
    
    if not orig_has_data and not trans_has_data:
        # No data available for either model
        plt.text(0.5, 0.5, f"No {metric_name} data available for plotting", 
                ha='center', va='center', fontsize=14)
        plt.gca().set_frame_on(False)
        plt.gca().set_xticks([])
        plt.gca().set_yticks([])
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Empty plot saved to {save_path}")
        
        return plt.gcf()
    
    # Plot data for original model if available
    if orig_has_data:
        # Get data to plot
        steps = original_results["step_history"]
        metrics = original_results[metric_key]
        
        # Highlight step 0 with a different marker
        has_step_0 = 0 in steps
        
        if has_step_0:
            idx_0 = steps.index(0)
            
            # Plot step 0 separately with a different marker
            plt.plot(
                [0], 
                [metrics[idx_0]], 
                marker='*', 
                markersize=12,
                linestyle='none',
                color='blue', 
                label=f'Original Model (Initial)'
            )
            
            # Plot the rest of the points
            if len(steps) > 1:
                # Check if we should connect initial point to first step
                if connect_initial:
                    # Draw a line from step 0 to first step
                    plt.plot(
                        steps[:2], 
                        metrics[:2], 
                        linestyle='-', 
                        color='blue',
                        alpha=0.7
                    )
                    
                    # Plot the rest excluding step 0
                    if len(steps) > 2:
                        plt.plot(
                            steps[1:], 
                            metrics[1:], 
                            marker='o', 
                            linestyle='-', 
                            color='blue', 
                            label='Original Model (Fine-tuned)'
                        )
                else:
                    # Plot all steps after step 0 without connecting to step 0
                    plt.plot(
                        steps[1:], 
                        metrics[1:], 
                        marker='o', 
                        linestyle='-', 
                        color='blue', 
                        label='Original Model (Fine-tuned)'
                    )
        else:
            # Plot all points normally if no step 0
            plt.plot(
                steps, 
                metrics, 
                marker='o', 
                linestyle='-', 
                color='blue', 
                label='Original Model'
            )
        
        # Add annotation for final point
        # final_step = steps[-1]
        # final_metric = metrics[-1]
        
        # plt.annotate(
        #     f"Final: {final_metric:.2f}", 
        #     xy=(final_step, final_metric),
        #     xytext=(final_step-5, final_metric+0.1),
        #     arrowprops=dict(facecolor='blue', shrink=0.05, width=1.5, headwidth=8),
        #     fontsize=10,
        #     color='blue'
        # )
        
        # Add annotation for initial point if available
        if has_step_0:
            init_metric = metrics[idx_0]
            plt.annotate(
                f"Initial: {init_metric:.2f}", 
                xy=(0, init_metric),
                xytext=(2, init_metric+0.1),
                arrowprops=dict(facecolor='blue', shrink=0.05, width=1.5, headwidth=8),
                fontsize=10,
                color='blue'
            )
    
    # Plot data for transformed model if available
    if trans_has_data:
        # Get data to plot
        steps = transformed_results["step_history"]
        metrics = transformed_results[metric_key]
        
        # Highlight step 0 with a different marker
        has_step_0 = 0 in steps
        
        if has_step_0:
            idx_0 = steps.index(0)
            
            # Plot step 0 separately with a different marker
            plt.plot(
                [0], 
                [metrics[idx_0]], 
                marker='*', 
                markersize=12,
                linestyle='none',
                color='green', 
                label='Transformed Model (Initial)'
            )
            
            # Plot the rest of the points
            if len(steps) > 1:
                # Check if we should connect initial point to first step
                if connect_initial:
                    # Draw a line from step 0 to first step
                    plt.plot(
                        steps[:2], 
                        metrics[:2], 
                        linestyle='-', 
                        color='green',
                        alpha=0.7
                    )
                    
                    # Plot the rest excluding step 0
                    if len(steps) > 2:
                        plt.plot(
                            steps[1:], 
                            metrics[1:], 
                            marker='s', 
                            linestyle='-', 
                            color='green', 
                            label='Transformed Model (Fine-tuned)'
                        )
                else:
                    # Plot all steps after step 0 without connecting to step 0
                    plt.plot(
                        steps[1:], 
                        metrics[1:], 
                        marker='s', 
                        linestyle='-', 
                        color='green', 
                        label='Transformed Model (Fine-tuned)'
                    )
        else:
            # Plot all points normally if no step 0
            plt.plot(
                steps, 
                metrics, 
                marker='s', 
                linestyle='-', 
                color='green', 
                label='Transformed Model'
            )
        
        # Add annotation for final point
        # final_step = steps[-1]
        # final_metric = metrics[-1]
        
        # plt.annotate(
        #     f"Final: {final_metric:.2f}", 
        #     xy=(final_step, final_metric),
        #     xytext=(final_step-5, final_metric-0.1),
        #     arrowprops=dict(facecolor='green', shrink=0.05, width=1.5, headwidth=8),
        #     fontsize=10,
        #     color='green'
        # )
        
        # Add annotation for initial point if available
        if has_step_0:
            init_metric = metrics[idx_0]
            plt.annotate(
                f"Initial: {init_metric:.2f}", 
                xy=(0, init_metric),
                xytext=(2, init_metric-0.1),
                arrowprops=dict(facecolor='green', shrink=0.05, width=1.5, headwidth=8),
                fontsize=10,
                color='green'
            )
    
    # Add labels and legend
    plt.xlabel("Training Steps")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add a note if one model didn't have data
    if not orig_has_data:
        plt.figtext(0.5, 0.01, "Note: No data available for Original Model", 
                   ha='center', fontsize=10, color='red')
    elif not trans_has_data:
        plt.figtext(0.5, 0.01, "Note: No data available for Transformed Model", 
                   ha='center', fontsize=10, color='red')
    
    # Save figure if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    return plt.gcf()

def load_results(original_file, transformed_file):
    """Load results from JSON files."""
    
    # Initialize with empty data
    original_results = {"perplexity_history": [], "ce_loss_history": [], "step_history": []}
    transformed_results = {"perplexity_history": [], "ce_loss_history": [], "step_history": []}
    
    # Load original model results if file exists
    if os.path.exists(original_file):
        try:
            with open(original_file, "r") as f:
                original_results = json.load(f)
            print(f"Loaded original model results from {original_file}")
            # Add empty ce_loss_history if not present in older format
            if "ce_loss_history" not in original_results:
                original_results["ce_loss_history"] = []
        except Exception as e:
            print(f"Error loading original model results: {e}")
    else:
        print(f"Warning: Original model results file {original_file} not found")
    
    # Load transformed model results if file exists
    if os.path.exists(transformed_file):
        try:
            with open(transformed_file, "r") as f:
                transformed_results = json.load(f)
            print(f"Loaded transformed model results from {transformed_file}")
            # Add empty ce_loss_history if not present in older format
            if "ce_loss_history" not in transformed_results:
                transformed_results["ce_loss_history"] = []
        except Exception as e:
            print(f"Error loading transformed model results: {e}")
    else:
        print(f"Warning: Transformed model results file {transformed_file} not found")
    
    return original_results, transformed_results

def main():
    # File paths
    original_file = "original_model_results.json"
    transformed_file = "transformed_model_results.json"
    perplexity_csv = "perplexity_comparison.csv"
    ce_loss_csv = "ce_loss_comparison.csv"
    perplexity_plot = "perplexity_vs_steps.png"
    ce_loss_plot = "ce_loss_vs_steps.png"
    combined_plot = "combined_metrics.png"
    
    # Try to load the results
    original_results, transformed_results = load_results(original_file, transformed_file)
    
    # Check if we have any data
    has_perplexity_data = (len(original_results.get("perplexity_history", [])) > 0 or 
                          len(transformed_results.get("perplexity_history", [])) > 0)
    
    has_ce_loss_data = (len(original_results.get("ce_loss_history", [])) > 0 or 
                        len(transformed_results.get("ce_loss_history", [])) > 0)
    
    if not has_perplexity_data and not has_ce_loss_data:
        print("No training data found in either file. Exiting.")
        return
    
    # Plot perplexity if available
    if has_perplexity_data:
        print("Plotting perplexity comparison...")
        plot_training_results(
            original_results, 
            transformed_results, 
            metric_name="perplexity",
            title="Llama-3.2-1B: Perplexity vs. Training Steps (WikiText-2)",
            save_path=perplexity_plot,
            connect_initial=True
        )
    
    # Plot CE loss if available
    if has_ce_loss_data:
        print("Plotting cross-entropy loss comparison...")
        plot_training_results(
            original_results, 
            transformed_results, 
            metric_name="ce_loss",
            title="Llama-3.2-1B: Cross-Entropy Loss vs. Training Steps (WikiText-2)",
            save_path=ce_loss_plot,
            connect_initial=True
        )
    
    # Create a combined plot with two subplots if both metrics are available
    if has_perplexity_data and has_ce_loss_data:
        print("Creating combined metrics plot...")
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
        
        # Plot perplexity on the first subplot
        plt.sca(ax1)
        plot_training_results(
            original_results, 
            transformed_results, 
            metric_name="perplexity",
            title="Perplexity vs. Training Steps",
            connect_initial=True
        )
        
        # Plot CE loss on the second subplot
        plt.sca(ax2)
        plot_training_results(
            original_results, 
            transformed_results, 
            metric_name="ce_loss",
            title="Cross-Entropy Loss vs. Training Steps",
            connect_initial=True
        )
        
        # Adjust layout and save
        plt.tight_layout()
        fig.suptitle("Llama-3.2-1B Model Performance (WikiText-2)", fontsize=16, y=1.02)
        plt.savefig(combined_plot, dpi=300, bbox_inches='tight')
        print(f"Combined plot saved to {combined_plot}")
    
    # Create DataFrames with aligned steps for fair comparison
    all_steps = sorted(list(set(
        original_results.get("step_history", []) + 
        transformed_results.get("step_history", [])
    )))
    
    # Process perplexity data
    if has_perplexity_data:
        orig_perplexity = []
        trans_perplexity = []
        
        for step in all_steps:
            # Find closest steps in original model results
            if step in original_results.get("step_history", []):
                idx = original_results["step_history"].index(step)
                orig_perplexity.append(original_results["perplexity_history"][idx])
            else:
                orig_perplexity.append(None)
                
            # Find closest steps in transformed model results
            if step in transformed_results.get("step_history", []):
                idx = transformed_results["step_history"].index(step)
                trans_perplexity.append(transformed_results["perplexity_history"][idx])
            else:
                trans_perplexity.append(None)
        
        # Calculate percent difference where both values are available
        perplexity_diff_list = []
        for i, (orig, trans) in enumerate(zip(orig_perplexity, trans_perplexity)):
            if orig is not None and trans is not None:
                pct_diff = ((trans - orig) / orig) * 100
                perplexity_diff_list.append(f"{pct_diff:.2f}%")
            else:
                perplexity_diff_list.append(None)
        
        # Save perplexity results to CSV
        perplexity_df = pd.DataFrame({
            "Step": all_steps,
            "Original Model Perplexity": orig_perplexity,
            "Transformed Model Perplexity": trans_perplexity,
            "Percent Difference": perplexity_diff_list
        })
        
        perplexity_df.to_csv(perplexity_csv, index=False)
        print(f"Perplexity results saved to {perplexity_csv}")
    
    # Process CE loss data
    if has_ce_loss_data:
        orig_ce_loss = []
        trans_ce_loss = []
        
        for step in all_steps:
            # Find closest steps in original model results
            if step in original_results.get("step_history", []) and len(original_results.get("ce_loss_history", [])) > 0:
                idx = original_results["step_history"].index(step)
                if idx < len(original_results["ce_loss_history"]):
                    orig_ce_loss.append(original_results["ce_loss_history"][idx])
                else:
                    orig_ce_loss.append(None)
            else:
                orig_ce_loss.append(None)
                
            # Find closest steps in transformed model results
            if step in transformed_results.get("step_history", []) and len(transformed_results.get("ce_loss_history", [])) > 0:
                idx = transformed_results["step_history"].index(step)
                if idx < len(transformed_results["ce_loss_history"]):
                    trans_ce_loss.append(transformed_results["ce_loss_history"][idx])
                else:
                    trans_ce_loss.append(None)
            else:
                trans_ce_loss.append(None)
        
        # Calculate percent difference where both values are available
        ce_loss_diff_list = []
        for i, (orig, trans) in enumerate(zip(orig_ce_loss, trans_ce_loss)):
            if orig is not None and trans is not None:
                pct_diff = ((trans - orig) / orig) * 100
                ce_loss_diff_list.append(f"{pct_diff:.2f}%")
            else:
                ce_loss_diff_list.append(None)
        
        # Save CE loss results to CSV
        ce_loss_df = pd.DataFrame({
            "Step": all_steps,
            "Original Model CE Loss": orig_ce_loss,
            "Transformed Model CE Loss": trans_ce_loss,
            "Percent Difference": ce_loss_diff_list
        })
        
        ce_loss_df.to_csv(ce_loss_csv, index=False)
        print(f"CE loss results saved to {ce_loss_csv}")
    
    # Print summary statistics
    print("\n=== SUMMARY STATISTICS ===")
    
    # Step 0 (Initial) comparison if available
    if 0 in original_results.get("step_history", []) and 0 in transformed_results.get("step_history", []):
        orig_idx_0 = original_results["step_history"].index(0)
        trans_idx_0 = transformed_results["step_history"].index(0)
        
        # Perplexity comparison
        if has_perplexity_data:
            orig_initial_ppl = original_results["perplexity_history"][orig_idx_0]
            trans_initial_ppl = transformed_results["perplexity_history"][trans_idx_0]
            
            init_ppl_diff_pct = ((trans_initial_ppl - orig_initial_ppl) / orig_initial_ppl) * 100
            
            print("\nInitial Perplexity Comparison (Step 0, before fine-tuning):")
            print(f"Original Model: {orig_initial_ppl:.4f}")
            print(f"Transformed Model: {trans_initial_ppl:.4f}")
            print(f"Difference: {init_ppl_diff_pct:+.2f}%")
        
        # CE loss comparison
        if has_ce_loss_data and orig_idx_0 < len(original_results.get("ce_loss_history", [])) and trans_idx_0 < len(transformed_results.get("ce_loss_history", [])):
            orig_initial_ce = original_results["ce_loss_history"][orig_idx_0]
            trans_initial_ce = transformed_results["ce_loss_history"][trans_idx_0]
            
            init_ce_diff_pct = ((trans_initial_ce - orig_initial_ce) / orig_initial_ce) * 100
            
            print("\nInitial CE Loss Comparison (Step 0, before fine-tuning):")
            print(f"Original Model: {orig_initial_ce:.4f}")
            print(f"Transformed Model: {trans_initial_ce:.4f}")
            print(f"Difference: {init_ce_diff_pct:+.2f}%")
    
    # Final metrics comparison
    if has_perplexity_data and original_results.get("perplexity_history") and transformed_results.get("perplexity_history"):
        orig_final_ppl = original_results["perplexity_history"][-1]
        trans_final_ppl = transformed_results["perplexity_history"][-1]
        final_ppl_diff_pct = ((trans_final_ppl - orig_final_ppl) / orig_final_ppl) * 100
        
        print("\nFinal Perplexity Comparison:")
        print(f"Original Model: {orig_final_ppl:.4f}")
        print(f"Transformed Model: {trans_final_ppl:.4f}")
        print(f"Difference: {final_ppl_diff_pct:+.2f}%")
    
    if has_ce_loss_data and original_results.get("ce_loss_history") and transformed_results.get("ce_loss_history"):
        orig_final_ce = original_results["ce_loss_history"][-1]
        trans_final_ce = transformed_results["ce_loss_history"][-1]
        final_ce_diff_pct = ((trans_final_ce - orig_final_ce) / orig_final_ce) * 100
        
        print("\nFinal CE Loss Comparison:")
        print(f"Original Model: {orig_final_ce:.4f}")
        print(f"Transformed Model: {trans_final_ce:.4f}")
        print(f"Difference: {final_ce_diff_pct:+.2f}%")
    
    # Include test mode fallback for demo purposes
    if not has_perplexity_data and not has_ce_loss_data:
        print("\nNo real data available. Creating demo plots with mock data...")
        
        mock_original = {
            "perplexity_history": [25.5, 24.8, 24.0, 23.5, 23.2],
            "ce_loss_history": [3.24, 3.21, 3.18, 3.16, 3.14],
            "step_history": [0, 10, 20, 30, 40]
        }
        
        mock_transformed = {
            "perplexity_history": [25.8, 24.9, 24.3, 23.8, 23.4],
            "ce_loss_history": [3.25, 3.22, 3.19, 3.17, 3.15],
            "step_history": [0, 10, 20, 30, 40]
        }
        
        # Create mock perplexity plot
        plot_training_results(
            mock_original, 
            mock_transformed, 
            metric_name="perplexity",
            title="Llama-3.2-1B: Mock Perplexity Data (Demo Only)",
            save_path="mock_perplexity_plot.png",
            connect_initial=True
        )
        
        # Create mock CE loss plot
        plot_training_results(
            mock_original, 
            mock_transformed, 
            metric_name="ce_loss",
            title="Llama-3.2-1B: Mock CE Loss Data (Demo Only)",
            save_path="mock_ce_loss_plot.png",
            connect_initial=True
        )
        
        print("Mock visualizations created as mock_perplexity_plot.png and mock_ce_loss_plot.png")

if __name__ == "__main__":
    main()