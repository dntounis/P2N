import os
import json
import random
import argparse
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.stats import gaussian_kde

def random_style(ax):
    if random.random() > 0.5:
        ax.set_facecolor(random.choice(['#f4f6f9', '#f9f6f4', '#f4f9f4', '#ffffff', '#eef2f5']))
    if random.random() > 0.3:
        ax.grid(True, linestyle=random.choice(['-', '--', ':', '-.']), alpha=random.uniform(0.2, 0.7))
    if random.random() > 0.3:
        ax.set_title("Plot " + "".join(random.choices("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789", k=5)))
    if random.random() > 0.3:
        ax.set_xlabel(random.choice(["X Axis", "Time", "Distance", "Energy [GeV]", "Mass", "Variable X", "Component 1"]))
    if random.random() > 0.3:
        ax.set_ylabel(random.choice(["Y Axis", "Events", "Cross Section", "Probability", "Value", "Variable Y", "Component 2"]))

def generate_scatter(ax):
    num_points = random.randint(5, 20)
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
    return [{"type": "scatter", "x": x, "y": y, "x_log": x_log, "y_log": y_log} for x, y in zip(x_vals, y_vals)]

def generate_fit(ax):
    num_points = random.randint(8, 25)
    x_vals = np.linspace(0, 10, num_points)
    a = random.uniform(-5.0, 5.0)
    b = random.uniform(0.0, 20.0)
    y_true = a * x_vals + b
    y_err = np.abs(np.random.normal(1.0, 0.5, num_points))
    y_obs = y_true + np.random.normal(0, y_err)
    
    with_error_bars = random.random() > 0.3
    if with_error_bars:
        ax.errorbar(x_vals, y_obs, yerr=y_err, fmt='ko', capsize=3, label='Data')
    else:
        ax.plot(x_vals, y_obs, 'ko', label='Data')
        
    ax.plot(x_vals, y_true, 'r-', label='Fit')
    
    if random.random() > 0.3:
        textstr = '\n'.join((
            r'$y = a x + b$',
            r'$a=%.2f \pm %.2f$' % (a, abs(a)*0.1),
            r'$b=%.2f \pm %.2f$' % (b, abs(b)*0.1)))
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=props)
    
    random_style(ax)
    if random.random() > 0.5:
        ax.legend()
        
    gt = [{"type": "fit_data", "x": round(float(x), 2), "y": round(float(y), 2)} for x, y in zip(x_vals, y_obs)]
    gt.append({"type": "fit_params", "a": round(a, 3), "b": round(b, 3)})
    return gt

def generate_clustering(ax):
    num_clusters = random.randint(2, 4)
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    gt = []
    
    for i in range(num_clusters):
        cx = random.uniform(0, 10)
        cy = random.uniform(0, 10)
        n_points = random.randint(20, 100)
        x = np.random.normal(cx, random.uniform(0.5, 1.5), n_points)
        y = np.random.normal(cy, random.uniform(0.5, 1.5), n_points)
        
        cluster_name = f"Cluster {chr(65+i)}"
        ax.scatter(x, y, color=colors[i], label=cluster_name, alpha=0.7, s=20)
        gt.append({"type": "cluster", "name": cluster_name, "centroid_x": round(cx, 2), "centroid_y": round(cy, 2), "size": n_points})
        
    ax.legend()
    random_style(ax)
    return gt

def generate_bar(ax):
    num_bars = random.randint(3, 10)
    x_vals = list(range(num_bars))
    y_vals = [round(random.uniform(10.0, 100.0), 1) for _ in range(num_bars)]
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#9b59b6', '#f1c40f', '#34495e']
    ax.bar(x_vals, y_vals, color=random.choice(colors), alpha=random.uniform(0.6, 1.0))
    random_style(ax)
    return [{"type": "bar", "category": str(x), "value": y} for x, y in zip(x_vals, y_vals)]

def generate_grouped_bar(ax):
    num_groups = random.randint(2, 4)
    num_categories = random.randint(2, 4)
    
    bar_width = 0.8 / num_categories
    x = np.arange(num_groups)
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    gt = []
    
    for i in range(num_categories):
        y_vals = [round(random.uniform(10, 100), 1) for _ in range(num_groups)]
        y_err = [round(random.uniform(2, 15), 1) for _ in range(num_groups)]
        
        offset = (i - num_categories/2) * bar_width + bar_width/2
        cat_name = f"Cat {i+1}"
        ax.bar(x + offset, y_vals, width=bar_width, yerr=y_err, label=cat_name, color=colors[i], capsize=3)
        
        for g in range(num_groups):
            gt.append({"type": "grouped_bar", "group": f"Group {g+1}", "category": cat_name, "value": y_vals[g], "error": y_err[g]})
            
    ax.set_xticks(x)
    ax.set_xticklabels([f"Group {g+1}" for g in range(num_groups)])
    ax.legend()
    random_style(ax)
    return gt

def generate_boxplot(ax):
    num_boxes = random.randint(3, 6)
    data = [np.random.normal(loc=random.uniform(10, 50), scale=random.uniform(5, 15), size=random.randint(20, 50)) for _ in range(num_boxes)]
    
    bp = ax.boxplot(data, patch_artist=True)
    colors = ['#ff9999','#66b3ff','#99ff99','#ffcc99','#c2c2f0','#ffb3e6']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
        
    ax.set_xticklabels([f"Set {i+1}" for i in range(num_boxes)])
    random_style(ax)
    
    gt = []
    for i, d in enumerate(data):
        q1, median, q3 = np.percentile(d, [25, 50, 75])
        gt.append({"type": "boxplot", "label": f"Set {i+1}", "q1": round(q1, 2), "median": round(median, 2), "q3": round(q3, 2)})
    return gt

def generate_pie(ax):
    num_slices = random.randint(3, 7)
    vals = [round(random.uniform(10.0, 100.0), 1) for _ in range(num_slices)]
    labels = [f"Class {i}" for i in range(num_slices)]
    ax.pie(vals, labels=labels if random.random() > 0.5 else None, autopct='%1.1f%%' if random.random() > 0.5 else None)
    if random.random() > 0.3:
        ax.set_title("Distribution " + "".join(random.choices("ABC", k=2)))
    return [{"type": "pie", "label": l, "value": v} for l, v in zip(labels, vals)]

def generate_histogram(ax):
    num_dists = random.randint(1, 2)
    bins = random.randint(10, 30)
    colors = ['blue', 'orange']
    gt = []
    
    for i in range(num_dists):
        data = np.random.normal(loc=random.uniform(20, 80), scale=random.uniform(5, 20), size=random.randint(200, 1000))
        counts, bin_edges, _ = ax.hist(data, bins=bins, color=colors[i], alpha=0.6, label=f"Dist {i+1}")
        
        for j in range(len(counts)):
            gt.append({"type": "histogram_bin", "dist": f"Dist {i+1}", "bin_start": round(float(bin_edges[j]), 2), 
                       "bin_end": round(float(bin_edges[j+1]), 2), "count": int(counts[j])})
            
    if num_dists > 1:
        ax.legend()
    random_style(ax)
    return gt

def generate_density(ax):
    num_dists = random.randint(2, 3)
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    gt = []
    x_grid = np.linspace(0, 100, 500)
    
    for i in range(num_dists):
        mu = random.uniform(20, 80)
        std = random.uniform(5, 15)
        data = np.random.normal(loc=mu, scale=std, size=1000)
        
        kde = gaussian_kde(data)
        density = kde(x_grid)
        
        ax.plot(x_grid, density, color=colors[i], linewidth=2, label=f"Dist {i+1}")
        ax.fill_between(x_grid, density, alpha=0.3, color=colors[i])
        
        # Add vertical line at mean
        ax.axvline(mu, color=colors[i], linestyle='--')
        
        # Add text box
        ax.text(mu, max(density)*1.05, f"{mu:.1f}", color='black', ha='center',
                bbox=dict(facecolor='white', edgecolor=colors[i], boxstyle='round,pad=0.2'))
                
        gt.append({"type": "density", "dist": f"Dist {i+1}", "mean": round(mu, 2), "std": round(std, 2)})
        
    ax.legend()
    random_style(ax)
    return gt

def generate_hep_brazil(ax):
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

def generate_heatmap(ax):
    data = np.random.randn(20, 20)
    # Add a pattern
    x = np.linspace(-3, 3, 20)
    y = np.linspace(-3, 3, 20)
    X, Y = np.meshgrid(x, y)
    Z = np.sin(X) * np.cos(Y) + data * 0.5
    
    cax = ax.imshow(Z, cmap=random.choice(['viridis', 'plasma', 'coolwarm', 'RdBu_r']), interpolation='nearest')
    plt.colorbar(cax, ax=ax, label="Function Value")
    random_style(ax)
    
    return [{"type": "heatmap", "min_z": round(float(np.min(Z)), 2), "max_z": round(float(np.max(Z)), 2)}]

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
    
    return [{"type": "contour_summary", "min_z": round(float(np.min(Z)), 2), "max_z": round(float(np.max(Z)), 2)}]

def generate_plot(output_dir, num_samples):
    os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
    metadata_path = os.path.join(output_dir, "metadata.jsonl")
    
    plot_types = [
        generate_scatter, generate_fit, generate_clustering, 
        generate_bar, generate_grouped_bar, generate_boxplot, 
        generate_pie, generate_histogram, generate_density,
        generate_hep_brazil, generate_heatmap, generate_contour
    ]
    
    with open(metadata_path, 'w') as f:
        for i in tqdm(range(num_samples), desc="Generating Complex Data"):
            fig, ax = plt.subplots(figsize=(random.uniform(5.5, 8.0), random.uniform(5.0, 7.0)))
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
