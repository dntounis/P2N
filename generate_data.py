import os
import json
import random
import argparse
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def random_style(ax):
    if random.random() > 0.5:
        ax.set_facecolor(random.choice(['#f4f6f9', '#f9f6f4', '#f4f9f4', '#ffffff', '#eef2f5']))
    if random.random() > 0.3:
        ax.grid(True, linestyle=random.choice(['-', '--', ':', '-.']), alpha=random.uniform(0.2, 0.7))
    if random.random() > 0.3:
        ax.set_title("Plot " + "".join(random.choices("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789", k=5)))
    if random.random() > 0.3:
        ax.set_xlabel(random.choice(["X Axis", "Time", "Distance", "Energy [GeV]", "Mass"]))
    if random.random() > 0.3:
        ax.set_ylabel(random.choice(["Y Axis", "Events", "Cross Section", "Probability", "Value"]))

def generate_scatter(ax):
    num_points = random.randint(5, 20)
    
    # Decide if axes should be log scale to generate appropriate random data ranges
    x_log = random.random() > 0.6
    y_log = random.random() > 0.6
    
    if x_log:
        x_vals = [round(10 ** random.uniform(-1.0, 3.0), 2) for _ in range(num_points)]
        ax.set_xscale('log')
    else:
        x_vals = [round(random.uniform(0.0, 100.0), 1) for _ in range(num_points)]
        
    if y_log:
        y_vals = [round(10 ** random.uniform(-1.0, 3.0), 2) for _ in range(num_points)]
        ax.set_yscale('log')
    else:
        y_vals = [round(random.uniform(0.0, 100.0), 1) for _ in range(num_points)]
        
    colors = ['blue', 'red', 'green', 'purple', 'orange', 'black', 'teal']
    markers = ['o', 's', '^', 'D', 'v', 'p', 'x', '+']
    ax.scatter(x_vals, y_vals, color=random.choice(colors), marker=random.choice(markers), 
               alpha=random.uniform(0.5, 1.0), s=random.uniform(20, 120))
    random_style(ax)
    
    # Store scale metadata so ground truth includes if it was log scale
    return [{"type": "scatter", "x": x, "y": y, "x_log": x_log, "y_log": y_log} for x, y in zip(x_vals, y_vals)]

def generate_bar(ax):
    num_bars = random.randint(3, 10)
    x_vals = list(range(num_bars))
    y_vals = [round(random.uniform(10.0, 100.0), 1) for _ in range(num_bars)]
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#9b59b6', '#f1c40f', '#34495e']
    ax.bar(x_vals, y_vals, color=random.choice(colors), alpha=random.uniform(0.6, 1.0))
    random_style(ax)
    return [{"type": "bar", "category": str(x), "value": y} for x, y in zip(x_vals, y_vals)]

def generate_pie(ax):
    num_slices = random.randint(3, 7)
    vals = [round(random.uniform(10.0, 100.0), 1) for _ in range(num_slices)]
    labels = [f"Class {i}" for i in range(num_slices)]
    ax.pie(vals, labels=labels if random.random() > 0.5 else None, autopct='%1.1f%%' if random.random() > 0.5 else None)
    if random.random() > 0.3:
        ax.set_title("Distribution " + "".join(random.choices("ABC", k=2)))
    return [{"type": "pie", "label": l, "value": v} for l, v in zip(labels, vals)]

def generate_histogram(ax):
    data = np.random.normal(loc=random.uniform(20, 80), scale=random.uniform(5, 20), size=random.randint(100, 1000))
    bins = random.randint(5, 20)
    counts, bin_edges, _ = ax.hist(data, bins=bins, color=random.choice(['blue', 'green', 'orange', 'grey']), alpha=0.7)
    random_style(ax)
    gt = []
    for i in range(len(counts)):
        gt.append({"type": "histogram_bin", "bin_start": round(float(bin_edges[i]), 2), 
                   "bin_end": round(float(bin_edges[i+1]), 2), "count": int(counts[i])})
    return gt

def generate_hep_brazil(ax):
    # High Energy Physics style brazil plot (expected limits with 1 and 2 sigma bands)
    masses = np.linspace(100, 1000, 10)
    expected = 10000 * masses ** -2 + np.random.normal(0, 0.05, len(masses))
    sigma1_up = expected * 1.5
    sigma1_down = expected * 0.6
    sigma2_up = expected * 2.2
    sigma2_down = expected * 0.3
    observed = expected * random.uniform(0.8, 1.2) + np.random.normal(0, 0.1, len(masses))
    
    ax.fill_between(masses, sigma2_down, sigma2_up, color='yellow', label='Expected 2$\sigma$')
    ax.fill_between(masses, sigma1_down, sigma1_up, color='lime', label='Expected 1$\sigma$')
    ax.plot(masses, expected, 'k--', label='Expected')
    if random.random() > 0.3:
        ax.plot(masses, observed, 'k-o', label='Observed')
    
    ax.set_yscale('log')
    ax.set_xlabel('Mass [GeV]')
    ax.set_ylabel('Limit')
    ax.legend()
    
    gt = []
    for i in range(len(masses)):
        gt.append({"type": "hep_limit", "mass": round(float(masses[i]), 1), "expected": round(float(expected[i]), 4),
                   "sigma1_up": round(float(sigma1_up[i]), 4), "sigma1_down": round(float(sigma1_down[i]), 4)})
    return gt

def generate_contour(ax):
    x = np.linspace(-3.0, 3.0, 100)
    y = np.linspace(-3.0, 3.0, 100)
    X, Y = np.meshgrid(x, y)
    Z1 = np.exp(-X**2 - Y**2)
    Z2 = np.exp(-(X - 1)**2 - (Y - 1)**2)
    Z = (Z1 - Z2) * 2
    
    if random.random() > 0.5:
        cs = ax.contourf(X, Y, Z, cmap=random.choice(['viridis', 'plasma', 'coolwarm', 'magma']))
    else:
        cs = ax.contour(X, Y, Z, cmap=random.choice(['viridis', 'plasma', 'coolwarm']))
        ax.clabel(cs, inline=1, fontsize=10)
    random_style(ax)
    
    # Ground truth is tricky for full contours, we will output the min/max or specific peaks
    return [{"type": "contour_summary", "min_z": round(float(np.min(Z)), 2), "max_z": round(float(np.max(Z)), 2)}]

def generate_plot(output_dir, num_samples):
    os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
    metadata_path = os.path.join(output_dir, "metadata.jsonl")
    
    plot_types = [generate_scatter, generate_bar, generate_pie, generate_histogram, generate_hep_brazil, generate_contour]
    
    with open(metadata_path, 'w') as f:
        for i in tqdm(range(num_samples), desc="Generating Complex Data"):
            fig, ax = plt.subplots(figsize=(random.uniform(5.0, 7.0), random.uniform(5.0, 7.0)))
            generator = random.choice(plot_types)
            plot_type_name = generator.__name__.replace('generate_', '')
            
            data_points = generator(ax)
            
            image_dir = os.path.join(output_dir, "images", plot_type_name)
            os.makedirs(image_dir, exist_ok=True)
            
            image_filename = f"image_{i:05d}.png"
            image_path = os.path.join(image_dir, image_filename)
            
            plt.savefig(image_path, bbox_inches='tight', dpi=random.randint(80, 150))
            plt.close(fig)
            
            ground_truth = {"gt_parse": {"data": data_points}}
            metadata_entry = {
                "file_name": f"images/{plot_type_name}/{image_filename}",
                "ground_truth": json.dumps(ground_truth)
            }
            f.write(json.dumps(metadata_entry) + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic scientific plots.")
    parser.add_argument("--output_dir", type=str, default="data", help="Output directory")
    parser.add_argument("--samples", type=int, default=100, help="Number of samples to generate")
    args = parser.parse_args()
    
    generate_plot(args.output_dir, args.samples)
    print(f"Generated {args.samples} samples in {args.output_dir}")
