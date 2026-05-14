import os
import io
import json
import random
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg') # Safe for multiprocessing
import matplotlib.pyplot as plt
import multiprocessing
from tqdm import tqdm
from scipy.stats import gaussian_kde
from matplotlib.patches import Ellipse
from PIL import Image, ImageFilter, ImageEnhance, ImageDraw, ImageFont


def degrade_image(image_path):
    """Apply realistic degradation effects to simulate scanned/old/low-quality plots.
    Randomly applies 1-3 effects from: blur, noise, JPEG artifacts, yellowing,
    rotation/skew, low contrast, low resolution, watermark, scanner edge shadow.
    Returns the degradation description for metadata."""
    img = Image.open(image_path).convert('RGB')
    applied = []

    # Pick 1-3 random degradation effects
    effects = random.sample([
        'blur', 'noise', 'jpeg_artifact', 'yellowing', 'rotation',
        'low_contrast', 'low_resolution', 'watermark', 'edge_shadow',
        'grayscale', 'brightness_shift'
    ], k=random.randint(1, 3))

    for effect in effects:
        if effect == 'blur':
            # Simulate out-of-focus scan or low-quality camera
            radius = random.uniform(0.5, 2.5)
            img = img.filter(ImageFilter.GaussianBlur(radius=radius))
            applied.append(f'blur_r{radius:.1f}')

        elif effect == 'noise':
            # Gaussian noise simulating old scanner / photocopy
            arr = np.array(img, dtype=np.float32)
            noise_level = random.uniform(5, 25)
            noise = np.random.normal(0, noise_level, arr.shape)
            arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
            img = Image.fromarray(arr)
            applied.append(f'noise_s{noise_level:.0f}')

        elif effect == 'jpeg_artifact':
            # Heavy JPEG compression artifacts
            quality = random.randint(8, 35)
            buffer = io.BytesIO()
            img.save(buffer, format='JPEG', quality=quality)
            buffer.seek(0)
            img = Image.open(buffer).convert('RGB')
            applied.append(f'jpeg_q{quality}')

        elif effect == 'yellowing':
            # Simulate aged/yellowed paper
            arr = np.array(img, dtype=np.float32)
            yellow_tint = np.array([random.uniform(10, 30), random.uniform(8, 20), -random.uniform(5, 20)])
            arr += yellow_tint
            arr = np.clip(arr, 0, 255).astype(np.uint8)
            img = Image.fromarray(arr)
            applied.append('yellowing')

        elif effect == 'rotation':
            # Slight skew from imperfect scan alignment
            angle = random.uniform(-3, 3)
            img = img.rotate(angle, resample=Image.BILINEAR, expand=False,
                           fillcolor=(255, 255, 255))
            applied.append(f'rot_{angle:.1f}deg')

        elif effect == 'low_contrast':
            # Faded/washed-out photocopy
            factor = random.uniform(0.3, 0.7)
            img = ImageEnhance.Contrast(img).enhance(factor)
            applied.append(f'contrast_{factor:.2f}')

        elif effect == 'low_resolution':
            # Simulate low-DPI scan or small embedded image
            scale = random.uniform(0.25, 0.5)
            w, h = img.size
            small = img.resize((int(w*scale), int(h*scale)), Image.BILINEAR)
            img = small.resize((w, h), Image.NEAREST)  # pixelated upscale
            applied.append(f'lowres_{scale:.2f}')

        elif effect == 'watermark':
            # Simulate institutional watermark or "DRAFT" stamp
            draw = ImageDraw.Draw(img)
            text = random.choice(['DRAFT', 'PREPRINT', 'CONFIDENTIAL', 'SAMPLE', 'COPY'])
            w, h = img.size
            try:
                font = ImageFont.truetype('/System/Library/Fonts/Helvetica.ttc', size=int(min(w,h)*0.15))
            except (OSError, IOError):
                font = ImageFont.load_default()
            # Semi-transparent diagonal stamp
            overlay = Image.new('RGBA', img.size, (0, 0, 0, 0))
            overlay_draw = ImageDraw.Draw(overlay)
            overlay_draw.text((w//4, h//3), text, fill=(200, 200, 200, 80), font=font)
            overlay = overlay.rotate(random.uniform(-30, -15), expand=False)
            img = Image.alpha_composite(img.convert('RGBA'), overlay).convert('RGB')
            applied.append(f'watermark_{text}')

        elif effect == 'edge_shadow':
            # Simulate scanner edge darkening (vignette)
            arr = np.array(img, dtype=np.float32)
            h, w = arr.shape[:2]
            Y, X = np.ogrid[:h, :w]
            cx, cy = w/2, h/2
            dist = np.sqrt((X-cx)**2 + (Y-cy)**2)
            max_dist = np.sqrt(cx**2 + cy**2)
            vignette = 1 - 0.4 * (dist / max_dist) ** 2
            arr *= vignette[:, :, np.newaxis]
            arr = np.clip(arr, 0, 255).astype(np.uint8)
            img = Image.fromarray(arr)
            applied.append('edge_shadow')

        elif effect == 'grayscale':
            # Old B&W photocopy
            img = img.convert('L').convert('RGB')
            applied.append('grayscale')

        elif effect == 'brightness_shift':
            # Uneven exposure from scanner lamp
            factor = random.uniform(0.6, 1.4)
            img = ImageEnhance.Brightness(img).enhance(factor)
            applied.append(f'brightness_{factor:.2f}')

    img.save(image_path)
    return applied

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

def generate_scatter(fig, ax):
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

def generate_fit(fig, ax):
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
    if random.random() > 0.5 and ax.get_legend_handles_labels()[1]: ax.legend(loc=random.choice(['upper left', 'upper right', 'lower left', 'lower right']))
        
    gt = [{"type": "fit_data", "x": round(float(x), 2), "y": round(float(y), 2)} for x, y in zip(x_vals, y_obs)]
    gt.append({"type": "fit_params", "a": round(a, 3), "b": round(b, 3)})
    return gt

def generate_clustering(fig, ax):
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
        
    if ax.get_legend_handles_labels()[1]: ax.legend(loc=random.choice(['upper left', 'upper right', 'lower left', 'lower right']))
    random_style(ax)
    return gt

def generate_bar(fig, ax):
    num_bars = random.randint(3, 10)
    x_vals = list(range(num_bars))
    y_vals = [round(random.uniform(10.0, 100.0), 1) for _ in range(num_bars)]
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#9b59b6', '#f1c40f', '#34495e']
    ax.bar(x_vals, y_vals, color=random.choice(colors), alpha=random.uniform(0.6, 1.0))
    random_style(ax)
    return [{"type": "bar", "category": str(x), "value": y} for x, y in zip(x_vals, y_vals)]

def generate_grouped_bar(fig, ax):
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
    if ax.get_legend_handles_labels()[1]: ax.legend(loc=random.choice(['upper left', 'upper right', 'lower left', 'lower right']))
    random_style(ax)
    return gt

def generate_boxplot(fig, ax):
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

def generate_pie(fig, ax):
    num_slices = random.randint(3, 7)
    vals = [round(random.uniform(10.0, 100.0), 1) for _ in range(num_slices)]
    labels = [f"Class {i}" for i in range(num_slices)]
    ax.pie(vals, labels=labels if random.random() > 0.5 else None, autopct='%1.1f%%' if random.random() > 0.5 else None)
    if random.random() > 0.3:
        ax.set_title("Distribution " + "".join(random.choices("ABC", k=2)))
    return [{"type": "pie", "label": l, "value": v} for l, v in zip(labels, vals)]

def generate_histogram(fig, ax):
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
            
    if num_dists > 1 and ax.get_legend_handles_labels()[1]: ax.legend(loc=random.choice(['upper left', 'upper right', 'lower left', 'lower right']))
    random_style(ax)
    return gt

def generate_density(fig, ax):
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
        
    if ax.get_legend_handles_labels()[1]: ax.legend(loc=random.choice(['upper left', 'upper right', 'lower left', 'lower right']))
    random_style(ax)
    return gt

def generate_hep_brazil(fig, ax):
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
    if ax.get_legend_handles_labels()[1]: ax.legend(loc=random.choice(['upper left', 'upper right', 'lower left', 'lower right']))
    
    gt = []
    for i in range(len(masses)):
        gt.append({"type": "hep_limit", "mass": round(float(masses[i]), 1), "expected": round(float(expected[i]), 4),
                   "sigma1_up": round(float(sigma1_up[i]), 4), "sigma1_down": round(float(sigma1_down[i]), 4)})
    return gt

def generate_heatmap(fig, ax):
    data = np.random.randn(20, 20)
    x = np.linspace(-3, 3, 20)
    y = np.linspace(-3, 3, 20)
    X, Y = np.meshgrid(x, y)
    Z = np.sin(X) * np.cos(Y) + data * 0.5
    
    cax = ax.imshow(Z, cmap=random.choice(['viridis', 'plasma', 'coolwarm', 'RdBu_r']), interpolation='nearest')
    plt.colorbar(cax, ax=ax, label="Function Value")
    random_style(ax)
    
    return [{"type": "heatmap", "min_z": round(float(np.min(Z)), 2), "max_z": round(float(np.max(Z)), 2)}]

def generate_contour(fig, ax):
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

def generate_corner_plot(fig, _):
    fig.clf() # We need a custom grid for a corner plot
    n_params = 3
    axes = fig.subplots(n_params, n_params)
    
    means = [random.uniform(0, 10) for _ in range(n_params)]
    stds = [random.uniform(0.5, 2.0) for _ in range(n_params)]
    names = [r"$\alpha$", r"$\beta$", r"$\gamma$"]
    
    gt = []
    
    for i in range(n_params):
        for j in range(n_params):
            ax = axes[i, j]
            if i < j:
                ax.axis('off') # Hide upper triangle
            elif i == j:
                # 1D marginal
                x = np.linspace(means[i]-3*stds[i], means[i]+3*stds[i], 100)
                y = np.exp(-0.5*((x-means[i])/stds[i])**2)
                ax.plot(x, y, 'r-')
                ax.set_yticks([])
                if i == 0:
                    ax.set_title(f"{names[i]} = {means[i]:.2f} $\pm$ {stds[i]:.2f}")
            else:
                # 2D contour
                x = np.linspace(means[j]-3*stds[j], means[j]+3*stds[j], 50)
                y = np.linspace(means[i]-3*stds[i], means[i]+3*stds[i], 50)
                X, Y = np.meshgrid(x, y)
                Z = np.exp(-0.5*(((X-means[j])/stds[j])**2 + ((Y-means[i])/stds[i])**2 + 0.5*(X-means[j])*(Y-means[i])/(stds[i]*stds[j])))
                ax.contourf(X, Y, Z, levels=[0.1, 0.5, 0.9], colors=['#ffcccc', '#ff9999', '#ff6666'], alpha=0.8)
                ax.contour(X, Y, Z, levels=[0.1, 0.5, 0.9], colors=['r', 'r', 'r'])
            
            if i == n_params - 1 and j != i:
                ax.set_xlabel(names[j])
            elif i != n_params - 1:
                ax.set_xticks([])
                
            if j == 0 and i != 0:
                ax.set_ylabel(names[i])
            elif j != 0:
                ax.set_yticks([])

    plt.subplots_adjust(wspace=0.05, hspace=0.05)
    
    for i in range(n_params):
        gt.append({"type": "corner_param", "name": names[i], "mean": round(means[i], 3), "std": round(stds[i], 3)})
    return gt

def generate_contour_overlay(fig, ax):
    # Cosmological constraints style
    x = np.linspace(0, 1, 100)
    y = np.linspace(0, 1.4, 100)
    X, Y = np.meshgrid(x, y)
    Z = (X-0.3)**2 + (Y-0.7)**2 + 0.8*(X-0.3)*(Y-0.7)
    
    ax.contour(X, Y, Z, levels=[0.05, 0.15, 0.3, 0.5], colors='black')
    
    # Overlaid points
    px = np.random.normal(0.3, 0.05, 50)
    py = np.random.normal(0.7, 0.05, 50)
    ax.scatter(px, py, color='orange', s=10, alpha=0.5)
    
    # Flat universe line
    ax.plot([0, 1], [1, 0], 'k-', label="Flat")
    
    ax.set_xlabel(r"$\Omega_M$")
    ax.set_ylabel(r"$\Omega_\Lambda$")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.4)
    random_style(ax)
    
    return [{"type": "contour_overlay", "center_x": 0.3, "center_y": 0.7}]

def generate_bump_hunt(fig, ax):
    x = np.linspace(100, 160, 40)
    bkg = 5000 * np.exp(-0.02 * x)
    sig = 200 * np.exp(-0.5 * ((x - 125) / 2)**2)
    obs = np.random.poisson(bkg + sig)
    
    ax.errorbar(x, obs, yerr=np.sqrt(obs), fmt='ko', label='Data')
    ax.plot(x, bkg, 'r--', label='Bkg Fit')
    ax.plot(x, bkg + sig, 'r-', label='S+B Fit')
    
    ax.fill_between(x, bkg*0.9, bkg*1.1, color='yellow', alpha=0.5, label='$\pm 1\sigma$')
    ax.fill_between(x, bkg*0.8, bkg*1.2, color='green', alpha=0.3, label='$\pm 2\sigma$')
    
    # Inset
    axins = ax.inset_axes([0.5, 0.5, 0.4, 0.4])
    axins.errorbar(x, obs, yerr=np.sqrt(obs), fmt='ko')
    axins.plot(x, bkg+sig, 'r-')
    axins.set_xlim(115, 135)
    axins.set_ylim(min(obs[(x>115)&(x<135)])*0.9, max(obs[(x>115)&(x<135)])*1.1)
    
    ax.set_xlabel(r"$m_{\gamma\gamma}$ (GeV)")
    ax.set_ylabel("Events / 1.5 GeV")
    if ax.get_legend_handles_labels()[1]: ax.legend(loc=random.choice(['upper left', 'upper right', 'lower left', 'lower right']))
    
    return [{"type": "bump_hunt", "peak": 125, "sig_yield": 200}]

def generate_stacked_ratio(fig, _):
    fig.clf()
    gs = fig.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    
    bins = np.arange(5)
    bkg1 = np.random.uniform(5, 15, 4)
    bkg2 = np.random.uniform(2, 8, 4)
    bkg3 = np.random.uniform(1, 5, 4)
    
    total_bkg = bkg1 + bkg2 + bkg3
    data = np.random.poisson(total_bkg)
    
    bottom = np.zeros(4)
    ax1.bar(bins[:-1], bkg1, align='edge', width=1, bottom=bottom, label='Z', color='#3182bd')
    bottom += bkg1
    ax1.bar(bins[:-1], bkg2, align='edge', width=1, bottom=bottom, label='ttbar', color='#e6550d')
    bottom += bkg2
    ax1.bar(bins[:-1], bkg3, align='edge', width=1, bottom=bottom, label='W', color='#756bb1')
    
    ax1.errorbar(bins[:-1]+0.5, data, yerr=np.sqrt(data), fmt='ko', label='Data')
    
    # Error bands
    for i in range(4):
        ax1.fill_between([i, i+1], bottom[i]*0.9, bottom[i]*1.1, hatch='//', facecolor='none', edgecolor='grey')
    
    ax1.legend()
    ax1.set_ylabel('Events')
    
    # Ratio plot
    ratio = data / total_bkg
    ratio_err = np.sqrt(data) / total_bkg
    ax2.errorbar(bins[:-1]+0.5, ratio, yerr=ratio_err, fmt='ko')
    ax2.axhline(1, color='k', linestyle='--')
    ax2.fill_between([0, 4], 0.9, 1.1, hatch='//', facecolor='none', edgecolor='grey')
    ax2.set_ylabel('Data / Model')
    ax2.set_xlabel('Signal Region')
    ax2.set_xticks(bins[:-1]+0.5)
    ax2.set_xticklabels(['SR1', 'SR2', 'SR3', 'SR4'])
    
    gt = []
    for i in range(4):
        gt.append({"type": "stacked_ratio", "bin": f"SR{i+1}", "data": int(data[i]), "bkg_total": round(total_bkg[i], 2)})
    return gt

def generate_double_y_axis(fig, ax):
    x = np.arange(10, 39)
    y1 = np.random.uniform(0, 8, len(x)) * 1e34  # Peak luminosity
    y2 = np.cumsum(np.random.uniform(10, 200, len(x)))  # Integrated luminosity
    
    ax.plot(x, y1, 'ro', markersize=4, label='Peak luminosity')
    ax.set_ylabel('Luminosity [cm$^{-2}$s$^{-1}$]', color='r')
    ax.tick_params(axis='y', labelcolor='r')
    
    ax2 = ax.twinx()
    ax2.plot(x, y2, 'b-', linewidth=2, label='Integrated luminosity')
    ax2.set_ylabel('Integrated luminosity [fb$^{-1}$]', color='b')
    ax2.tick_params(axis='y', labelcolor='b')
    
    # Add shaded regions for LS1, LS2 etc.
    for i in range(1, 6):
        ax.axvspan(10 + i*5, 10 + i*5 + 2, color='blue', alpha=0.2)
        ax.text(10 + i*5 + 1, 4e34, f'LS{i}', rotation=90, color='white', fontweight='bold', ha='center')
        
    ax.set_xlabel('Year')
    fig.legend(loc="upper center", ncol=2)
    
    gt = []
    for i in range(len(x)):
        gt.append({"type": "double_y", "year": int(x[i]), "peak": float(y1[i]), "integrated": float(y2[i])})
    return gt

def generate_multi_line_log(fig, ax):
    x = np.logspace(0, 2, 100)
    y1 = 10 * x**-1.5 + np.random.normal(0, 0.1, len(x))
    y2 = 5 * x**-1.2 + np.random.normal(0, 0.05, len(x))
    
    ax.plot(x, y1, 'r-', label='1st gen.')
    ax.plot(x, y2, 'gray', label='Next gen.', linestyle='--')
    
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'Primary mass ($M_\odot$)')
    ax.set_ylabel(r'$\Gamma (Gpc^{-3} yr^{-1} M_\odot^{-1})$')
    if ax.get_legend_handles_labels()[1]: ax.legend(loc=random.choice(['upper left', 'upper right', 'lower left', 'lower right']))
    
    # Annotations
    ax.annotate('The cliff', xy=(40, 10**-1), xytext=(50, 10**-0.5), color='blue',
                arrowprops=dict(facecolor='blue', shrink=0.05))
    
    gt = [{"type": "multi_line", "x_min": round(float(x.min()), 2), "x_max": round(float(x.max()), 2)}]
    return gt

def generate_stacked_histogram(fig, ax):
    bins = np.linspace(70, 500, 30)
    
    bkg1 = np.random.normal(90, 5, 500)  # Z peak
    bkg2 = np.random.normal(200, 30, 800) # Broad background
    sig = np.random.normal(125, 2, 50)   # Higgs
    
    ax.hist([bkg1, bkg2, sig], bins=bins, stacked=True, 
            color=['#99ccff', '#6699ff', '#ff9999'], 
            label=['Z+X', 'ZZ', 'H(125)'])
            
    # Data
    counts, _ = np.histogram(np.concatenate([bkg1, bkg2, sig]), bins=bins)
    data = np.random.poisson(counts)
    bin_centers = 0.5 * (bins[1:] + bins[:-1])
    ax.errorbar(bin_centers, data, yerr=np.sqrt(data), fmt='ko', label='Data')
    
    ax.set_xlabel(r'$m_{4\ell}$ (GeV)')
    ax.set_ylabel('Events / 4 GeV')
    if ax.get_legend_handles_labels()[1]: ax.legend(loc=random.choice(['upper left', 'upper right', 'lower left', 'lower right']))
    
    gt = [{"type": "stacked_hist", "peaks": [90, 125, 200]}]
    return gt

def generate_residual_bump(fig, ax):
    x = np.linspace(100, 160, 30)
    residual = np.random.normal(0, 2, len(x))
    
    # Add bump
    bump_idx = (x > 120) & (x < 130)
    residual[bump_idx] += 5 * np.exp(-0.5 * ((x[bump_idx] - 125) / 2)**2)
    
    ax.errorbar(x, residual, yerr=np.ones_like(x)*2, fmt='k^')
    ax.axhline(0, color='b', linestyle='--')
    
    # Fit line
    x_fit = np.linspace(100, 160, 100)
    y_fit = 6 * np.exp(-0.5 * ((x_fit - 125) / 1.5)**2)
    ax.plot(x_fit, y_fit, 'b-', linewidth=2)
    
    ax.text(125, -4, r'H$\to\gamma\gamma$', ha='center')
    
    ax.set_xlabel(r'$m_{\gamma\gamma}$ [GeV]')
    ax.set_ylabel('Residuals')
    
    gt = [{"type": "residual_bump", "peak_x": 125, "peak_y": 6}]
    return gt

def generate_ashby_chart(fig, ax):
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    materials = ['Polymers', 'Composites', 'Wood', 'Metals', 'Ceramics', 'Foams', 'Elastomers']
    colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99', '#c2c2f0', '#ffb3e6', '#c4e17f']
    
    gt = []
    for mat, color in zip(materials, colors):
        x_center = 10 ** random.uniform(1, 4)
        y_center = 10 ** random.uniform(-1, 3)
        width = x_center * random.uniform(0.5, 1.5)
        height = y_center * random.uniform(0.5, 1.5)
        angle = random.uniform(-45, 45)
        
        ellipse = Ellipse((x_center, y_center), width, height, angle=angle, 
                          facecolor=color, alpha=0.6, edgecolor='black')
        ax.add_patch(ellipse)
        ax.text(x_center, y_center, mat, ha='center', va='center', fontsize=8)
        
        gt.append({"type": "ashby_bubble", "material": mat, "x_center": round(x_center, 2), "y_center": round(y_center, 2)})
        
    ax.set_xlim(10, 10000)
    ax.set_ylim(0.01, 1000)
    ax.set_xlabel(r'Density (kg/m$^3$)')
    ax.set_ylabel('Young\'s Modulus (GPa)')
    ax.grid(True, which="both", ls="--", alpha=0.5)
    
    return gt

def generate_phase_diagram(fig, ax):
    T = np.linspace(200, 700, 100)
    
    # Triple point
    T_tp, P_tp = 273.16, 0.006
    # Critical point
    T_cp, P_cp = 647.096, 217.7
    
    # Solid-Gas
    T_sg = np.linspace(200, T_tp, 50)
    P_sg = P_tp * np.exp(6000 * (1/T_tp - 1/T_sg))
    
    # Liquid-Gas
    T_lg = np.linspace(T_tp, T_cp, 50)
    P_lg = P_tp * np.exp(4000 * (1/T_tp - 1/T_lg))
    
    # Solid-Liquid (anomalous for water: negative slope)
    T_sl = np.linspace(273.16, 260, 50)
    P_sl = np.linspace(P_tp, 1000, 50)
    
    ax.plot(T_sg, P_sg, 'r-')
    ax.plot(T_lg, P_lg, 'r-')
    ax.plot(T_sl, P_sl, 'r-')
    
    ax.plot([T_tp], [P_tp], 'ko')
    ax.text(T_tp, P_tp*0.5, 'Triple\nPoint', ha='center', va='top')
    
    ax.plot([T_cp], [P_cp], 'ko')
    ax.text(T_cp, P_cp*1.5, 'Critical\nPoint', ha='center', va='bottom')
    
    ax.text(240, 10, 'Ice', fontsize=12)
    ax.text(400, 10, 'Water', fontsize=12)
    ax.text(400, 0.001, 'Vapor', fontsize=12)
    
    ax.set_yscale('log')
    ax.set_xlim(200, 700)
    ax.set_ylim(0.0001, 1000)
    ax.set_xlabel('Temperature (K)')
    ax.set_ylabel('Pressure (atm)')
    ax.set_title('Phase Diagram of Water')
    
    gt = [{"type": "phase_diagram", "triple_point": [T_tp, P_tp], "critical_point": [T_cp, P_cp]}]
    return gt

def generate_parity_grid(fig, _):
    fig.clf()
    axes = fig.subplots(1, 3)
    models = ['M3GNet', 'TensorNet', 'CHGNet']
    colors = ['#ff7f0e', '#d62728', '#8c564b']
    
    gt = []
    for i, ax in enumerate(axes):
        x = np.random.uniform(0, 400, 500)
        noise = np.random.normal(0, 20 + i*5, 500)
        y = x + noise
        
        ax.scatter(x, y, color=colors[i], s=5, alpha=0.6)
        ax.plot([0, 400], [0, 400], 'k-', lw=1)
        
        r2 = 1 - np.var(y - x) / np.var(x)
        mae = np.mean(np.abs(y - x))
        ax.text(0.05, 0.95, f'$R^2$ = {r2:.2f}\nMAE = {mae:.2f}', transform=ax.transAxes, va='top')
        
        ax.set_xlabel(r'$K_{DFT}$ (GPa)')
        if i == 0:
            ax.set_ylabel(r'$K_{%s}$ (GPa)' % models[i])
        else:
            ax.set_ylabel(r'$K_{%s}$ (GPa)' % models[i])
            
        ax.set_xlim(0, 400)
        ax.set_ylim(0, 400)
        
        gt.append({"type": "parity_plot", "model": models[i], "R2": round(r2, 2), "MAE": round(mae, 2)})
        
    plt.tight_layout()
    return gt

def generate_stress_strain(fig, ax):
    strain = np.linspace(0, 0.3, 100)
    E = 200e3 # Elastic modulus
    yield_strain = 0.005
    yield_stress = E * yield_strain
    
    # Elastic region
    elastic = strain <= yield_strain
    stress = np.zeros_like(strain)
    stress[elastic] = E * strain[elastic]
    
    # Plastic region (Hollomon's equation: sigma = K * epsilon^n)
    K = 1500
    n = 0.2
    plastic = strain > yield_strain
    stress[plastic] = K * (strain[plastic])**n
    
    # Smooth transition
    smooth_idx = (strain > yield_strain - 0.002) & (strain < yield_strain + 0.005)
    stress[smooth_idx] = np.interp(strain[smooth_idx], 
                                   [strain[smooth_idx][0], strain[smooth_idx][-1]], 
                                   [stress[smooth_idx][0], stress[smooth_idx][-1]])
    
    ax.plot(strain, stress, 'b-', lw=2)
    
    # Annotations
    ax.plot([yield_strain], [yield_stress], 'bo')
    ax.annotate('Yield\nStrength', xy=(yield_strain, yield_stress), xytext=(yield_strain+0.02, yield_stress-100),
                arrowprops=dict(arrowstyle="->", color='gray'))
                
    uts_idx = np.argmax(stress)
    ax.plot([strain[uts_idx]], [stress[uts_idx]], 'bo')
    ax.annotate('Ultimate\nStrength', xy=(strain[uts_idx], stress[uts_idx]), xytext=(strain[uts_idx], stress[uts_idx]-150),
                arrowprops=dict(arrowstyle="->", color='gray'))
                
    ax.plot([strain[-1]], [stress[-1]], 'bo')
    ax.annotate('Fracture', xy=(strain[-1], stress[-1]), xytext=(strain[-1]-0.05, stress[-1]-100),
                arrowprops=dict(arrowstyle="->", color='gray'))
                
    # Regions
    ax.axvspan(0, yield_strain, color='green', alpha=0.2)
    ax.text(yield_strain/2, max(stress)*0.9, 'Elastic\nRegion', ha='center', va='center')
    ax.text(0.15, max(stress)*0.9, 'Plastic Region', ha='center', va='center')
    
    ax.set_xlabel(r'Strain, $\varepsilon$')
    ax.set_ylabel(r'Stress, $\sigma$')
    ax.set_xlim(0, 0.32)
    ax.set_ylim(0, max(stress)*1.1)
    
    gt = [{"type": "stress_strain", "yield_stress": float(yield_stress), "uts": float(stress[uts_idx])}]
    return gt

def generate_volcano_plot(fig, ax):
    n_genes = 500
    logFC = np.random.normal(0, 2, n_genes)
    p_vals = np.random.uniform(0, 1, n_genes)
    neg_log10_p = -np.log10(p_vals)
    
    # add some true positives
    logFC = np.append(logFC, np.random.normal(4, 1, 50))
    neg_log10_p = np.append(neg_log10_p, np.random.normal(5, 1, 50))
    logFC = np.append(logFC, np.random.normal(-4, 1, 50))
    neg_log10_p = np.append(neg_log10_p, np.random.normal(5, 1, 50))
    
    sig_up = (logFC > 2) & (neg_log10_p > 2)
    sig_down = (logFC < -2) & (neg_log10_p > 2)
    not_sig = ~(sig_up | sig_down)
    
    ax.scatter(logFC[not_sig], neg_log10_p[not_sig], color='gray', alpha=0.5, s=10)
    ax.scatter(logFC[sig_up], neg_log10_p[sig_up], color='red', alpha=0.7, s=15, label='Up')
    ax.scatter(logFC[sig_down], neg_log10_p[sig_down], color='blue', alpha=0.7, s=15, label='Down')
    
    ax.axvline(-2, color='k', linestyle='--')
    ax.axvline(2, color='k', linestyle='--')
    ax.axhline(2, color='k', linestyle='--')
    
    ax.set_xlabel(r'Log$_2$ Fold Change')
    ax.set_ylabel(r'-Log$_{10}$ P-value')
    if ax.get_legend_handles_labels()[1]: ax.legend(loc=random.choice(['upper left', 'upper right', 'lower left', 'lower right']))
    random_style(ax)
    
    return [{"type": "volcano_summary", "up_regulated": int(np.sum(sig_up)), "down_regulated": int(np.sum(sig_down))}]

def generate_roc_curve(fig, ax):
    x = np.linspace(0, 1, 50)
    models = ['Model A', 'Model B', 'Baseline']
    colors = ['r', 'b', 'k']
    
    gt = []
    for m, c in zip(models, colors):
        if m == 'Baseline':
            y = x
            auc = 0.50
            ls = '--'
        else:
            power = random.uniform(2, 5)
            y = x**(1/power)
            auc = 1 - (1 / (power + 1))
            ls = '-'
        
        ax.plot(x, y, color=c, linestyle=ls, label=f'{m} (AUC = {auc:.2f})')
        gt.append({"type": "roc_curve", "model": m, "auc": round(auc, 2)})
        
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    if ax.get_legend_handles_labels()[1]: ax.legend(loc=random.choice(['upper left', 'upper right', 'lower left', 'lower right']))
    random_style(ax)
    return gt

def generate_light_curve(fig, ax):
    time = np.linspace(59000, 59100, 60)
    # transient event
    mag = 20 - 5 * np.exp(-0.5 * ((time - 59050) / 5)**2) + np.random.normal(0, 0.2, len(time))
    err = np.random.uniform(0.1, 0.3, len(time))
    
    ax.errorbar(time, mag, yerr=err, fmt='o', color='green', markersize=4)
    
    ax.invert_yaxis() # Magnitudes are inverted
    ax.set_xlabel('Time (MJD)')
    ax.set_ylabel('Apparent Magnitude (g-band)')
    random_style(ax)
    
    return [{"type": "light_curve", "peak_time": 59050, "peak_mag": round(float(np.min(mag)), 2)}]


def generate_line_plot(fig, ax):
    n_series = random.randint(1, 4)
    colors = ['k', 'r', 'b', 'g']; styles = ['-', '--', '-.', ':']
    markers = ['', 'o', 's', '^']
    x = np.linspace(0, 10, random.randint(20, 80))
    gt = []
    for i in range(n_series):
        freq = random.uniform(0.5, 3); amp = random.uniform(1, 10)
        y = amp * np.sin(freq * x) + np.random.normal(0, 0.5, len(x))
        name = f"Series {i+1}"
        ax.plot(x, y, color=colors[i], linestyle=styles[i], marker=markers[i] if random.random()>0.5 else '', 
                markevery=max(1,len(x)//10), label=name)
        gt.append({"type": "line_series", "name": name, "x_min": round(float(x[0]),2), "x_max": round(float(x[-1]),2),
                   "y_mean": round(float(np.mean(y)),2)})
    if random.random() > 0.5: ax.set_xscale('log') if random.random()>0.7 else None
    if ax.get_legend_handles_labels()[1]: ax.legend(loc=random.choice(['upper left', 'upper right', 'lower left', 'lower right'])); ax.set_xlabel(random.choice(['Time (s)', 'Energy (eV)', 'Temperature (K)']))
    ax.set_ylabel(random.choice(['Signal', 'Intensity', 'Response']))
    return gt


def generate_violin_plot(fig, ax):
    n = random.randint(3, 6)
    data = [np.random.normal(random.uniform(20,60), random.uniform(5,15), random.randint(50,200)) for _ in range(n)]
    parts = ax.violinplot(data, showmeans=True, showmedians=True)
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(['#ff9999','#66b3ff','#99ff99','#ffcc99','#c2c2f0','#ffb3e6'][i])
        pc.set_alpha(0.7)
    ax.set_xticks(range(1, n+1)); ax.set_xticklabels([f"Group {i+1}" for i in range(n)])
    gt = []
    for i, d in enumerate(data):
        gt.append({"type": "violin", "group": f"Group {i+1}", "mean": round(float(np.mean(d)),2), 
                   "median": round(float(np.median(d)),2), "std": round(float(np.std(d)),2)})
    return gt


def generate_spatial_map(fig, ax):
    lon = np.linspace(-180, 180, 50); lat = np.linspace(-90, 90, 50)
    LON, LAT = np.meshgrid(lon, lat)
    Z = np.sin(np.radians(LON)) * np.cos(np.radians(LAT)) + np.random.normal(0, 0.1, LON.shape)
    c = ax.pcolormesh(LON, LAT, Z, cmap='RdBu_r', shading='auto')
    plt.colorbar(c, ax=ax, label='Anomaly (°C)')
    ax.set_xlabel('Longitude'); ax.set_ylabel('Latitude')
    return [{"type": "spatial_map", "z_min": round(float(Z.min()),2), "z_max": round(float(Z.max()),2)}]


def generate_invariant_mass(fig, _):
    fig.clf(); gs = fig.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0)
    ax1 = fig.add_subplot(gs[0]); ax2 = fig.add_subplot(gs[1], sharex=ax1)
    x = np.linspace(60, 200, 50); bkg = 500*np.exp(-0.03*x)
    sig = 300*np.exp(-0.5*((x-91)/2.5)**2)  # Z boson
    sig2 = 50*np.exp(-0.5*((x-125)/2)**2)   # Higgs
    total = bkg + sig + sig2
    data = np.random.poisson(total.astype(int))
    ax1.fill_between(x, 0, bkg, step='mid', color='#ffcc00', alpha=0.7, label='Bkg')
    ax1.fill_between(x, bkg, bkg+sig, step='mid', color='#3182bd', alpha=0.7, label='Z')
    ax1.fill_between(x, bkg+sig, total, step='mid', color='#e6550d', alpha=0.7, label='H')
    ax1.errorbar(x, data, yerr=np.sqrt(data), fmt='ko', ms=3, label='Data')
    ax1.set_ylabel('Events / 3 GeV'); ax1.legend(); ax1.set_yscale('log')
    ratio = data / total; ratio_err = np.sqrt(data) / total
    ax2.errorbar(x, ratio, yerr=ratio_err, fmt='ko', ms=3)
    ax2.axhline(1, color='r', ls='--'); ax2.set_ylabel('Data/MC'); ax2.set_xlabel(r'$m_{\ell\ell}$ [GeV]')
    ax2.set_ylim(0.5, 1.5)
    return [{"type": "invariant_mass", "z_peak": 91, "h_peak": 125}]


def generate_pt_spectrum(fig, ax):
    pt = np.logspace(1, 3, 40)
    mc = 1e6 * pt**-4.5; data_vals = np.random.poisson(mc.astype(int).clip(1))
    ax.errorbar(pt, data_vals, yerr=np.sqrt(data_vals), fmt='ko', ms=3, label='Data')
    ax.plot(pt, mc, 'r-', label='MC'); ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel(r'$p_T$ [GeV]'); ax.set_ylabel('Events')
    if ax.get_legend_handles_labels()[1]: ax.legend(loc=random.choice(['upper left', 'upper right', 'lower left', 'lower right']))
    return [{"type": "pt_spectrum", "pt_min": round(float(pt[0]),1), "pt_max": round(float(pt[-1]),1)}]


def generate_pull_plot(fig, ax):
    n_np = random.randint(10, 25)
    names = [f"NP_{i}" for i in range(n_np)]
    pulls = np.random.normal(0, 0.8, n_np); constraints = np.random.uniform(0.5, 1.2, n_np)
    y = np.arange(n_np)
    ax.errorbar(pulls, y, xerr=constraints, fmt='ko', capsize=3)
    ax.axvline(0, color='k', ls='-'); ax.axvline(1, color='r', ls='--'); ax.axvline(-1, color='r', ls='--')
    ax.axvspan(-1, 1, color='yellow', alpha=0.15); ax.axvspan(-2, 2, color='green', alpha=0.08)
    ax.set_yticks(y); ax.set_yticklabels(names, fontsize=7)
    ax.set_xlabel(r'Pull ($\sigma$)'); ax.set_xlim(-3, 3)
    return [{"type": "pull", "name": n, "pull": round(float(p),2), "constraint": round(float(c),2)} 
            for n, p, c in zip(names, pulls, constraints)]


def generate_correlation_matrix(fig, ax):
    n = random.randint(6, 12)
    A = np.random.randn(100, n); corr = np.corrcoef(A.T)
    labels = [f"p{i}" for i in range(n)]
    im = ax.imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1)
    plt.colorbar(im, ax=ax); ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=45, fontsize=7); ax.set_yticklabels(labels, fontsize=7)
    return [{"type": "correlation_matrix", "size": n, "min_corr": round(float(corr.min()),2)}]


def generate_unfolded_xsec(fig, _):
    fig.clf(); gs = fig.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0)
    ax1 = fig.add_subplot(gs[0]); ax2 = fig.add_subplot(gs[1], sharex=ax1)
    x = np.linspace(0, 500, 15); xsec = 100*np.exp(-0.005*x)
    err = xsec * np.random.uniform(0.05, 0.15, len(x))
    theory = 95*np.exp(-0.0048*x)
    ax1.errorbar(x, xsec, yerr=err, fmt='ko', label='Data')
    ax1.fill_between(x, theory*0.9, theory*1.1, color='red', alpha=0.3, label='Theory')
    ax1.plot(x, theory, 'r-'); ax1.set_ylabel(r'd$\sigma$/d$p_T$ [pb/GeV]'); ax1.set_yscale('log'); ax1.legend()
    ratio = xsec/theory; ratio_err = err/theory
    ax2.errorbar(x, ratio, yerr=ratio_err, fmt='ko'); ax2.axhline(1, color='r', ls='--')
    ax2.set_ylabel('Data/Theory'); ax2.set_xlabel(r'$p_T$ [GeV]'); ax2.set_ylim(0.5, 1.5)
    return [{"type": "unfolded_xsec", "n_bins": len(x)}]


def generate_efficiency_map(fig, ax):
    pt = np.linspace(20, 200, 15); eta = np.linspace(-2.5, 2.5, 10)
    PT, ETA = np.meshgrid(pt, eta)
    eff = 0.9 - 0.1*np.exp(-PT/50) - 0.05*ETA**2/6 + np.random.normal(0, 0.02, PT.shape)
    eff = np.clip(eff, 0, 1)
    c = ax.pcolormesh(PT, ETA, eff, cmap='viridis', vmin=0, vmax=1, shading='auto')
    plt.colorbar(c, ax=ax, label='Efficiency')
    ax.set_xlabel(r'$p_T$ [GeV]'); ax.set_ylabel(r'$\eta$')
    return [{"type": "efficiency_map", "mean_eff": round(float(eff.mean()),3)}]


def generate_sky_map(fig, _):
    fig.clf(); ax = fig.add_subplot(111, projection='mollweide')
    lon = np.linspace(-np.pi, np.pi, 100); lat = np.linspace(-np.pi/2, np.pi/2, 50)
    LON, LAT = np.meshgrid(lon, lat)
    # CMB-like fluctuations
    Z = sum(np.sin(i*LON)*np.cos(j*LAT)*random.uniform(-1,1) for i in range(1,6) for j in range(1,4))
    ax.pcolormesh(LON, LAT, Z, cmap='RdBu_r', shading='auto')
    ax.set_title('Sky Map'); ax.grid(True, alpha=0.3)
    return [{"type": "sky_map", "z_range": round(float(Z.max()-Z.min()),2)}]


def generate_sed(fig, ax):
    freq = np.logspace(8, 18, 50)  # Hz
    flux = 1e-23 * (freq/1e12)**-0.7 * (1 + 0.3*np.sin(np.log10(freq)))
    flux *= np.random.lognormal(0, 0.1, len(freq))
    ax.scatter(freq, flux, c='k', s=15, label='Data')
    ax.plot(freq, 1e-23*(freq/1e12)**-0.7, 'r-', label='Model')
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel('Frequency (Hz)'); ax.set_ylabel(r'Flux Density (Jy)')
    if ax.get_legend_handles_labels()[1]: ax.legend(loc=random.choice(['upper left', 'upper right', 'lower left', 'lower right']))
    return [{"type": "sed", "freq_min": round(float(freq[0]),1), "freq_max": round(float(freq[-1]),1)}]


def generate_hr_diagram(fig, ax):
    n = 500
    temp = 10**np.random.uniform(3.5, 4.5, n)  # K
    lum = 10**np.random.normal(0, 2, n)
    # main sequence
    ms_temp = 10**np.linspace(3.5, 4.5, 200)
    ms_lum = (ms_temp/5778)**4
    ax.scatter(temp, lum, c='gray', s=5, alpha=0.5)
    ax.plot(ms_temp, ms_lum, 'r-', lw=2, label='Main Sequence')
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.invert_xaxis()  # convention: hot stars left
    ax.set_xlabel('Temperature (K)'); ax.set_ylabel(r'Luminosity ($L_\odot$)')
    if ax.get_legend_handles_labels()[1]: ax.legend(loc=random.choice(['upper left', 'upper right', 'lower left', 'lower right']))
    return [{"type": "hr_diagram", "n_stars": n}]


def generate_power_spectrum(fig, ax):
    ell = np.arange(2, 2500)
    cl = 6000 / (ell * (ell+1)) * np.exp(-(ell/1500)**2) * (1 + 0.3*np.sin(ell/200))
    cl_err = cl * 0.05
    dll = ell*(ell+1)*cl / (2*np.pi)
    dll_err = ell*(ell+1)*cl_err / (2*np.pi)
    ax.errorbar(ell[::10], dll[::10], yerr=dll_err[::10], fmt='k.', ms=2, alpha=0.5)
    ax.plot(ell, dll, 'r-', label='Best fit')
    ax.set_xlabel(r'Multipole $\ell$'); ax.set_ylabel(r'$D_\ell$ [$\mu K^2$]')
    ax.set_xscale('log')
    if ax.get_legend_handles_labels()[1]: ax.legend(loc=random.choice(['upper left', 'upper right', 'lower left', 'lower right']))
    return [{"type": "power_spectrum", "ell_max": int(ell[-1])}]


def generate_redshift_distribution(fig, ax):
    z = np.linspace(0, 3, 40)
    n1 = 200*z**2*np.exp(-z/0.5); n2 = 100*z**2*np.exp(-z/0.8)
    ax.bar(z, n1, width=z[1]-z[0], alpha=0.6, color='blue', label='Photometric')
    ax.step(z, n2, color='red', lw=2, label='Spectroscopic')
    ax.set_xlabel('Redshift z'); ax.set_ylabel('N(z)')
    if ax.get_legend_handles_labels()[1]: ax.legend(loc=random.choice(['upper left', 'upper right', 'lower left', 'lower right']))
    return [{"type": "redshift_dist", "z_peak_phot": round(float(z[np.argmax(n1)]),2)}]


def generate_mass_radius(fig, ax):
    m = np.logspace(-1, 2, 80)
    r = m**0.8 * np.random.lognormal(0, 0.2, len(m))
    colors = np.random.choice(['red','blue','green','orange'], len(m))
    ax.scatter(m, r, c=colors, s=15, alpha=0.7)
    ax.plot(m, m**0.8, 'k--', label=r'$R \propto M^{0.8}$')
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel(r'Mass ($M_\odot$)'); ax.set_ylabel(r'Radius ($R_\odot$)')
    if ax.get_legend_handles_labels()[1]: ax.legend(loc=random.choice(['upper left', 'upper right', 'lower left', 'lower right']))
    return [{"type": "mass_radius", "n_objects": len(m)}]


def generate_residual_map(fig, ax):
    x = np.linspace(-5, 5, 60); y = np.linspace(-5, 5, 60)
    X, Y = np.meshgrid(x, y)
    Z = np.random.normal(0, 1, X.shape) + 3*np.exp(-((X-1)**2+(Y+1)**2)/2)
    c = ax.pcolormesh(X, Y, Z, cmap='RdBu_r', shading='auto', vmin=-4, vmax=4)
    ax.contour(X, Y, Z, levels=[2, 3, 4], colors='k', linewidths=0.5)
    plt.colorbar(c, ax=ax, label=r'Significance ($\sigma$)')
    ax.set_xlabel('RA offset (arcsec)'); ax.set_ylabel('Dec offset (arcsec)')
    return [{"type": "residual_map", "peak_sig": round(float(Z.max()),2)}]


def generate_band_structure(fig, ax):
    k = np.linspace(0, 3, 200)
    for i in range(6):
        E = -4 + i*1.5 + 0.8*np.sin(k*np.pi) + 0.3*np.cos(2*k*np.pi) + random.uniform(-0.5, 0.5)
        ax.plot(k, E, 'b-', lw=1)
    ax.axhline(0, color='r', ls='--', label=r'$E_F$')
    for xv, lbl in zip([0, 1, 2, 3], [r'$\Gamma$', 'X', 'M', r'$\Gamma$']):
        ax.axvline(xv, color='k', lw=0.5)
    ax.set_xticks([0, 1, 2, 3]); ax.set_xticklabels([r'$\Gamma$', 'X', 'M', r'$\Gamma$'])
    ax.set_ylabel('Energy (eV)'); ax.set_xlim(0, 3)
    if ax.get_legend_handles_labels()[1]: ax.legend(loc=random.choice(['upper left', 'upper right', 'lower left', 'lower right']))
    return [{"type": "band_structure", "n_bands": 6, "fermi_level": 0}]


def generate_dos(fig, ax):
    E = np.linspace(-8, 8, 300)
    total = sum(np.exp(-0.5*((E-c)/0.5)**2)*random.uniform(0.5,3) for c in np.random.uniform(-6,6,8))
    s_dos = total * 0.3; p_dos = total * 0.5; d_dos = total * 0.2
    ax.fill_betweenx(E, 0, s_dos, alpha=0.5, color='blue', label='s')
    ax.fill_betweenx(E, 0, p_dos, alpha=0.5, color='red', label='p')
    ax.fill_betweenx(E, 0, d_dos, alpha=0.5, color='green', label='d')
    ax.axhline(0, color='k', ls='--', label=r'$E_F$')
    ax.set_ylabel('Energy (eV)'); ax.set_xlabel('DOS (states/eV)')
    if ax.get_legend_handles_labels()[1]: ax.legend(loc=random.choice(['upper left', 'upper right', 'lower left', 'lower right']))
    return [{"type": "dos", "e_range": [-8, 8]}]


def generate_xrd(fig, ax):
    two_theta = np.linspace(10, 90, 500)
    intensity = np.zeros_like(two_theta) + 50
    peaks = sorted(np.random.uniform(15, 85, random.randint(5, 12)))
    gt = []
    for p in peaks:
        h = random.uniform(200, 5000); w = random.uniform(0.1, 0.5)
        intensity += h * np.exp(-0.5*((two_theta-p)/w)**2)
        gt.append({"type": "xrd_peak", "two_theta": round(p,2), "intensity": round(h,1)})
    ax.plot(two_theta, intensity, 'k-', lw=0.8)
    ax.set_xlabel(r'2$\theta$ (°)'); ax.set_ylabel('Intensity (a.u.)'); ax.set_xlim(10, 90)
    return gt


def generate_raman_spectrum(fig, ax):
    wavenumber = np.linspace(100, 3500, 500)
    intensity = np.zeros_like(wavenumber) + 100
    peaks = np.random.uniform(200, 3200, random.randint(4, 10))
    for p in peaks:
        intensity += random.uniform(100,2000) * np.exp(-0.5*((wavenumber-p)/random.uniform(5,30))**2)
    n_spectra = random.randint(1, 3)
    colors = ['k', 'r', 'b']
    for i in range(n_spectra):
        offset = i * 500
        ax.plot(wavenumber, intensity*(1+0.1*i) + offset + np.random.normal(0,20,len(wavenumber)), 
                color=colors[i], label=f'Sample {i+1}')
    ax.set_xlabel(r'Wavenumber (cm$^{-1}$)'); ax.set_ylabel('Intensity (a.u.)')
    if ax.get_legend_handles_labels()[1]: ax.legend(loc=random.choice(['upper left', 'upper right', 'lower left', 'lower right']))
    return [{"type": "raman", "n_peaks": len(peaks)}]


def generate_magnetization(fig, ax):
    H = np.linspace(-2, 2, 200)
    Ms = random.uniform(0.5, 2.0); Hc = random.uniform(0.1, 0.5)
    M_up = Ms * np.tanh((H - Hc) * 3); M_down = Ms * np.tanh((H + Hc) * 3)
    ax.plot(H, M_up, 'b-', label='Forward'); ax.plot(H, M_down, 'r-', label='Reverse')
    ax.axhline(0, color='k', lw=0.5); ax.axvline(0, color='k', lw=0.5)
    ax.set_xlabel('H (T)'); ax.set_ylabel(r'M ($\mu_B$/atom)')
    if ax.get_legend_handles_labels()[1]: ax.legend(loc=random.choice(['upper left', 'upper right', 'lower left', 'lower right']))
    return [{"type": "magnetization", "Ms": round(Ms,2), "Hc": round(Hc,2)}]


def generate_resistivity(fig, ax):
    T = np.linspace(2, 300, 200)
    Tc = random.uniform(4, 92)
    rho = np.where(T < Tc, 0, 0.1*(T-Tc) + 0.001*T**2)
    rho += np.random.normal(0, 0.5, len(T))
    rho = np.clip(rho, 0, None)
    ax.plot(T, rho, 'k-', lw=1.5)
    ax.axvline(Tc, color='r', ls='--', label=f'$T_c$ = {Tc:.0f} K')
    ax.set_xlabel('Temperature (K)'); ax.set_ylabel(r'$\rho$ (m$\Omega$ cm)')
    if ax.get_legend_handles_labels()[1]: ax.legend(loc=random.choice(['upper left', 'upper right', 'lower left', 'lower right']))
    return [{"type": "resistivity", "Tc": round(Tc,1)}]


def generate_nmr_spectrum(fig, ax):
    ppm = np.linspace(12, 0, 1000)  # decreasing convention
    intensity = np.zeros_like(ppm)
    peaks = np.random.uniform(0.5, 10, random.randint(4, 8))
    gt = []
    for p in peaks:
        h = random.uniform(0.5, 5); w = random.uniform(0.02, 0.08)
        intensity += h * np.exp(-0.5*((ppm-p)/w)**2)
        gt.append({"type": "nmr_peak", "ppm": round(p,2), "height": round(h,2)})
    ax.plot(ppm, intensity, 'k-', lw=0.8); ax.invert_xaxis()
    ax.set_xlabel('Chemical Shift (ppm)'); ax.set_ylabel('Intensity')
    return gt


def generate_mass_spectrum_chem(fig, ax):
    mz = np.arange(10, 300)
    intensities = np.zeros_like(mz, dtype=float)
    peak_positions = sorted(random.sample(range(20, 280), random.randint(8, 20)))
    gt = []
    for p in peak_positions:
        h = random.uniform(10, 100)
        intensities[p-10] = h
        gt.append({"type": "mass_peak", "mz": p, "intensity": round(h,1)})
    ax.stem(mz, intensities, linefmt='k-', markerfmt='', basefmt='k-')
    ax.set_xlabel('m/z'); ax.set_ylabel('Relative Abundance (%)')
    return gt


def generate_uv_vis(fig, ax):
    wl = np.linspace(200, 800, 300)
    n_samples = random.randint(1, 3)
    colors = ['b', 'r', 'g']; gt = []
    for i in range(n_samples):
        peaks = np.random.uniform(250, 600, random.randint(1, 3))
        absorbance = sum(random.uniform(0.3,2)*np.exp(-0.5*((wl-p)/random.uniform(15,40))**2) for p in peaks)
        ax.plot(wl, absorbance, color=colors[i], label=f'Sample {i+1}')
        gt.append({"type": "uv_vis", "sample": f"Sample {i+1}", "lambda_max": round(float(wl[np.argmax(absorbance)]),1)})
    ax.set_xlabel('Wavelength (nm)'); ax.set_ylabel('Absorbance')
    if ax.get_legend_handles_labels()[1]: ax.legend(loc=random.choice(['upper left', 'upper right', 'lower left', 'lower right']))
    return gt


def generate_chromatogram(fig, ax):
    t = np.linspace(0, 30, 500)
    signal = np.random.normal(0, 0.01, len(t))
    peaks = sorted(np.random.uniform(2, 28, random.randint(3, 8)))
    gt = []
    for p in peaks:
        h = random.uniform(0.5, 5); w = random.uniform(0.1, 0.5)
        signal += h * np.exp(-0.5*((t-p)/w)**2)
        gt.append({"type": "chrom_peak", "rt": round(p,2), "height": round(h,2)})
    ax.plot(t, signal, 'b-'); ax.fill_between(t, 0, signal, alpha=0.2)
    ax.set_xlabel('Retention Time (min)'); ax.set_ylabel('Detector Response')
    return gt


def generate_reaction_coordinate(fig, ax):
    states = ['R', 'TS1', 'I', 'TS2', 'P']
    energies = [0, random.uniform(15,35), random.uniform(-5,10), random.uniform(20,40), random.uniform(-15,5)]
    x = np.arange(len(states))
    # plateaus with smooth connections
    for i in range(len(states)):
        ax.plot([x[i]-0.3, x[i]+0.3], [energies[i], energies[i]], 'b-', lw=2)
        if i < len(states)-1:
            xc = np.linspace(x[i]+0.3, x[i+1]-0.3, 20)
            yc = np.interp(xc, [x[i]+0.3, x[i+1]-0.3], [energies[i], energies[i+1]])
            ax.plot(xc, yc, 'b--', lw=1)
    ax.set_xticks(x); ax.set_xticklabels(states)
    ax.set_ylabel('Energy (kcal/mol)'); ax.set_xlabel('Reaction Coordinate')
    return [{"type": "rxn_coord", "state": s, "energy": round(e,1)} for s, e in zip(states, energies)]


def generate_kinetic_trace(fig, ax):
    t = np.linspace(0, 100, 60)
    temps = [300, 320, 340]; colors = ['b', 'r', 'g']; gt = []
    for T, c in zip(temps, colors):
        k = 0.01 * np.exp(-(5000/T)); C0 = random.uniform(0.8, 1.2)
        C = C0 * np.exp(-k*t) + np.random.normal(0, 0.02, len(t))
        ax.plot(t, C, f'{c}o-', ms=3, label=f'{T} K')
        gt.append({"type": "kinetic", "temp": T, "k": round(k,5), "C0": round(C0,2)})
    ax.set_xlabel('Time (min)'); ax.set_ylabel('Concentration (M)')
    if ax.get_legend_handles_labels()[1]: ax.legend(loc=random.choice(['upper left', 'upper right', 'lower left', 'lower right']))
    return gt


def generate_titration_curve(fig, ax):
    V = np.linspace(0, 50, 200)
    Ve = random.uniform(20, 35)  # equivalence point
    pH = 2 + 12 / (1 + np.exp(-(V - Ve) * 0.5))
    pH += np.random.normal(0, 0.1, len(V))
    ax.plot(V, pH, 'b-', lw=2)
    ax.axvline(Ve, color='r', ls='--', label=f'Eq. pt = {Ve:.1f} mL')
    ax.set_xlabel('Volume of Titrant (mL)'); ax.set_ylabel('pH')
    if ax.get_legend_handles_labels()[1]: ax.legend(loc=random.choice(['upper left', 'upper right', 'lower left', 'lower right']))
    return [{"type": "titration", "equiv_vol": round(Ve,1)}]


def generate_calibration_curve(fig, ax):
    conc = np.array([0, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0])
    slope = random.uniform(0.5, 3.0); intercept = random.uniform(0, 0.5)
    signal = slope * conc + intercept + np.random.normal(0, 0.1, len(conc))
    ax.plot(conc, signal, 'ko', ms=6)
    fit_x = np.linspace(0, 10, 100); fit_y = slope*fit_x + intercept
    ax.plot(fit_x, fit_y, 'r-')
    r2 = 1 - np.var(signal - (slope*conc+intercept)) / np.var(signal)
    ax.text(0.05, 0.9, f'y = {slope:.2f}x + {intercept:.2f}\n$R^2$ = {r2:.4f}', 
            transform=ax.transAxes, bbox=dict(facecolor='wheat', alpha=0.5))
    ax.set_xlabel('Concentration (mg/L)'); ax.set_ylabel('Signal'); 
    return [{"type": "calibration", "slope": round(slope,3), "intercept": round(intercept,3), "R2": round(r2,4)}]


def generate_spectroscopy_2d(fig, ax):
    x = np.linspace(1500, 1700, 80); y = np.linspace(1500, 1700, 80)
    X, Y = np.meshgrid(x, y)
    Z = np.exp(-((X-1600)**2 + (Y-1620)**2)/(2*20**2)) - 0.5*np.exp(-((X-1650)**2+(Y-1580)**2)/(2*15**2))
    c = ax.contourf(X, Y, Z, levels=20, cmap='RdBu_r')
    plt.colorbar(c, ax=ax, label='Intensity')
    ax.set_xlabel(r'$\omega_1$ (cm$^{-1}$)'); ax.set_ylabel(r'$\omega_3$ (cm$^{-1}$)')
    return [{"type": "spectroscopy_2d", "peak_1": [1600, 1620], "peak_2": [1650, 1580]}]


def generate_clustered_heatmap(fig, _):
    fig.clf()
    n_genes = 20; n_samples = 8
    data = np.random.randn(n_genes, n_samples)
    # add some structure
    data[:5, :4] += 2; data[10:15, 4:] -= 2
    ax = fig.add_subplot(111)
    im = ax.imshow(data, cmap='RdBu_r', aspect='auto')
    plt.colorbar(im, ax=ax, label='Z-score')
    ax.set_xlabel('Samples'); ax.set_ylabel('Genes')
    ax.set_xticks(range(n_samples)); ax.set_xticklabels([f'S{i}' for i in range(n_samples)], fontsize=7)
    return [{"type": "clustered_heatmap", "n_genes": n_genes, "n_samples": n_samples}]


def generate_manhattan_plot(fig, ax):
    gt = []; offset = 0; colors_chr = ['#1f77b4', '#ff7f0e']
    for chrom in range(1, 23):
        n = random.randint(50, 150)
        pos = np.sort(np.random.uniform(0, 1e8, n)) + offset
        pvals = np.random.uniform(0, 1, n); logp = -np.log10(pvals)
        # add some hits
        if random.random() > 0.7:
            logp[random.randint(0, n-1)] = random.uniform(8, 15)
        ax.scatter(pos, logp, s=3, c=colors_chr[chrom%2], alpha=0.6)
        offset += 1e8
    ax.axhline(7.3, color='r', ls='--', label='Genome-wide significance')
    ax.set_xlabel('Genomic Position'); ax.set_ylabel(r'-log$_{10}$(p)')
    if ax.get_legend_handles_labels()[1]: ax.legend(fontsize=7, loc=random.choice(['upper left', 'upper right', 'lower left', 'lower right']))
    return [{"type": "manhattan", "n_chromosomes": 22, "threshold": 7.3}]


def generate_survival_curve(fig, ax):
    t = np.sort(np.random.exponential(50, 100))
    groups = {'Treatment': 0.7, 'Control': 1.0}
    colors_km = {'Treatment': 'blue', 'Control': 'red'}; gt = []
    for name, hazard in groups.items():
        times = np.sort(np.random.exponential(50/hazard, 80))
        surv = np.linspace(1, 0.1, len(times))
        ax.step(times, surv, where='post', color=colors_km[name], lw=2, label=name)
        # censor ticks
        censor_idx = np.random.choice(len(times), 10, replace=False)
        ax.plot(times[censor_idx], surv[censor_idx], '|', color=colors_km[name], ms=10)
        gt.append({"type": "survival", "group": name, "median_survival": round(float(np.median(times)),1)})
    ax.set_xlabel('Time (months)'); ax.set_ylabel('Survival Probability')
    if ax.get_legend_handles_labels()[1]: ax.legend(loc=random.choice(['upper left', 'upper right', 'lower left', 'lower right'])); ax.text(0.7, 0.9, 'p = 0.003', transform=ax.transAxes)
    return gt


def generate_dose_response(fig, ax):
    dose = np.logspace(-3, 2, 30)
    ic50 = 10**random.uniform(-1, 1); hill = random.uniform(0.8, 2.5)
    response = 100 / (1 + (dose/ic50)**hill) + np.random.normal(0, 3, len(dose))
    ax.semilogx(dose, response, 'ko', ms=4)
    fit_x = np.logspace(-3, 2, 200); fit_y = 100 / (1 + (fit_x/ic50)**hill)
    ax.plot(fit_x, fit_y, 'r-')
    ax.axhline(50, color='gray', ls='--'); ax.axvline(ic50, color='gray', ls='--')
    ax.text(ic50*1.5, 55, f'IC50 = {ic50:.2f}', fontsize=9)
    ax.set_xlabel('Concentration (µM)'); ax.set_ylabel('Viability (%)')
    return [{"type": "dose_response", "ic50": round(ic50,3), "hill": round(hill,2)}]


def generate_flow_cytometry(fig, ax):
    n = 2000
    pop1_x = np.random.lognormal(3, 0.5, n); pop1_y = np.random.lognormal(2, 0.5, n)
    pop2_x = np.random.lognormal(5, 0.4, 500); pop2_y = np.random.lognormal(5, 0.4, 500)
    ax.scatter(pop1_x, pop1_y, s=1, alpha=0.3, c='blue')
    ax.scatter(pop2_x, pop2_y, s=1, alpha=0.3, c='red')
    # gate
    ax.plot([50,50,1000,1000,50], [50,1000,1000,50,50], 'k-', lw=1.5)
    ax.text(200, 800, f'{500/(n+500)*100:.1f}%', fontsize=10)
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel('CD4-FITC'); ax.set_ylabel('CD8-PE')
    return [{"type": "flow_cytometry", "gated_pct": round(500/(n+500)*100,1)}]


def generate_forest_plot(fig, ax):
    n_studies = random.randint(6, 12)
    names = [f"Study {i+1}" for i in range(n_studies)]
    effects = np.random.normal(0.3, 0.5, n_studies)
    ci_low = effects - np.random.uniform(0.1, 0.6, n_studies)
    ci_high = effects + np.random.uniform(0.1, 0.6, n_studies)
    y = np.arange(n_studies)
    sizes = np.random.uniform(50, 200, n_studies)
    ax.scatter(effects, y, s=sizes, c='blue', zorder=3)
    ax.hlines(y, ci_low, ci_high, colors='blue')
    # pooled
    pooled = np.mean(effects)
    ax.axvline(pooled, color='red', ls='-', lw=2, label=f'Pooled: {pooled:.2f}')
    ax.axvline(0, color='k', ls='--')
    ax.set_yticks(y); ax.set_yticklabels(names, fontsize=7)
    ax.set_xlabel('Effect Size (95% CI)')
    if ax.get_legend_handles_labels()[1]: ax.legend(fontsize=7, loc=random.choice(['upper left', 'upper right', 'lower left', 'lower right']))
    gt = [{"type": "forest", "study": n, "effect": round(float(e),3), "ci": [round(float(l),3), round(float(h),3)]} 
          for n, e, l, h in zip(names, effects, ci_low, ci_high)]
    return gt


def generate_epidemic_curve(fig, ax):
    weeks = np.arange(52)
    cases = np.random.poisson(50, 52)
    # wave
    cases = cases + (300 * np.exp(-0.5*((weeks-20)/5)**2)).astype(int)
    ax.bar(weeks, cases, color='#e6550d', alpha=0.8, label='Cases')
    # rolling average
    ra = np.convolve(cases, np.ones(4)/4, mode='same')
    ax.plot(weeks, ra, 'k-', lw=2, label='4-week avg')
    ax.set_xlabel('Week'); ax.set_ylabel('Cases')
    if ax.get_legend_handles_labels()[1]: ax.legend(loc=random.choice(['upper left', 'upper right', 'lower left', 'lower right']))
    return [{"type": "epidemic", "peak_week": int(weeks[np.argmax(cases)]), "total": int(np.sum(cases))}]


def generate_bland_altman(fig, ax):
    n = 50
    m1 = np.random.uniform(50, 150, n); m2 = m1 + np.random.normal(2, 5, n)
    mean_m = (m1 + m2) / 2; diff = m1 - m2
    bias = np.mean(diff); loa = 1.96 * np.std(diff)
    ax.scatter(mean_m, diff, c='k', s=20, alpha=0.6)
    ax.axhline(bias, color='r', ls='-', label=f'Bias: {bias:.1f}')
    ax.axhline(bias+loa, color='b', ls='--', label=f'+1.96 SD: {bias+loa:.1f}')
    ax.axhline(bias-loa, color='b', ls='--', label=f'-1.96 SD: {bias-loa:.1f}')
    ax.set_xlabel('Mean of Methods'); ax.set_ylabel('Difference')
    if ax.get_legend_handles_labels()[1]: ax.legend(fontsize=7, loc=random.choice(['upper left', 'upper right', 'lower left', 'lower right']))
    return [{"type": "bland_altman", "bias": round(bias,2), "loa_upper": round(bias+loa,2), "loa_lower": round(bias-loa,2)}]


def generate_waterfall_plot(fig, ax):
    n = random.randint(20, 40)
    changes = np.random.uniform(-80, 60, n)
    changes = np.sort(changes)
    colors = ['green' if c < -30 else ('red' if c > 20 else 'gray') for c in changes]
    ax.bar(range(n), changes, color=colors)
    ax.axhline(-30, color='k', ls='--'); ax.axhline(20, color='k', ls='--')
    ax.set_xlabel('Patient'); ax.set_ylabel('% Change from Baseline')
    return [{"type": "waterfall", "n_responders": sum(1 for c in changes if c < -30), "n_patients": n}]


def generate_spaghetti_plot(fig, ax):
    n_patients = 30; n_visits = 8
    t = np.arange(n_visits)
    for i in range(n_patients):
        baseline = random.uniform(100, 200)
        trend = random.uniform(-5, 2)
        y = baseline + trend*t + np.random.normal(0, 10, n_visits)
        ax.plot(t, y, 'b-', alpha=0.15, lw=1)
    # group mean
    mean_y = 150 - 2*t; ax.plot(t, mean_y, 'r-', lw=3, label='Group Mean')
    ax.set_xlabel('Visit'); ax.set_ylabel('Biomarker Level')
    if ax.get_legend_handles_labels()[1]: ax.legend(loc=random.choice(['upper left', 'upper right', 'lower left', 'lower right']))
    return [{"type": "spaghetti", "n_patients": n_patients, "n_visits": n_visits}]


def generate_raster_plot(fig, ax):
    n_neurons=20; t_max=500
    for i in range(n_neurons):
        spikes = np.sort(np.random.uniform(0, t_max, random.randint(10,80)))
        ax.vlines(spikes, i+0.5, i+1.5, lw=0.5)
    ax.axvspan(200,300,color='yellow',alpha=0.3,label='Stimulus')
    ax.set_xlabel('Time (ms)'); ax.set_ylabel('Neuron'); ax.set_xlim(0,t_max)
    if ax.get_legend_handles_labels()[1]: ax.legend(fontsize=7, loc=random.choice(['upper left', 'upper right', 'lower left', 'lower right']))
    return [{"type":"raster","n_neurons":n_neurons,"t_max":t_max}]


def generate_psth(fig, ax):
    t=np.linspace(-200,500,70); fr=5+15*np.exp(-0.5*((t-100)/50)**2)+np.random.normal(0,1,len(t))
    fr=np.clip(fr,0,None)
    ax.bar(t, fr, width=t[1]-t[0], color='steelblue', alpha=0.7)
    ax.axvspan(0,200,color='yellow',alpha=0.2,label='Stimulus')
    ax.axvline(0,color='k',ls='--')
    ax.set_xlabel('Time from stimulus (ms)'); ax.set_ylabel('Firing rate (Hz)')
    if ax.get_legend_handles_labels()[1]: ax.legend(fontsize=7, loc=random.choice(['upper left', 'upper right', 'lower left', 'lower right']))
    return [{"type":"psth","peak_rate":round(float(fr.max()),1)}]


def generate_tuning_curve(fig, ax):
    angles=np.linspace(0,360,36); pref=random.uniform(0,360)
    fr=20+30*np.exp(-0.5*((angles-pref)/40)**2)+np.random.normal(0,2,len(angles))
    ax.errorbar(angles,fr,yerr=2,fmt='ko-',ms=3,capsize=2)
    ax.axvline(pref,color='r',ls='--',label=f'Pref={pref:.0f}°')
    ax.set_xlabel('Direction (°)'); ax.set_ylabel('Firing Rate (Hz)')
    if ax.get_legend_handles_labels()[1]: ax.legend(loc=random.choice(['upper left', 'upper right', 'lower left', 'lower right']))
    return [{"type":"tuning","preferred_dir":round(pref,1)}]


def generate_eeg_traces(fig, ax):
    t=np.linspace(0,1,500); channels=['Fz','Cz','Pz','Oz']
    for i,ch in enumerate(channels):
        sig=np.random.normal(0,10,500); sig+=5*np.sin(2*np.pi*10*t)
        ax.plot(t, sig+i*60, 'k-', lw=0.5)
        ax.text(-0.02, i*60, ch, ha='right', fontsize=8)
    ax.axvline(0.3,color='r',ls='--',label='Event')
    ax.set_xlabel('Time (s)'); ax.set_ylabel('Amplitude (µV)')
    if ax.get_legend_handles_labels()[1]: ax.legend(fontsize=7, loc=random.choice(['upper left', 'upper right', 'lower left', 'lower right']))
    return [{"type":"eeg","n_channels":len(channels)}]


def generate_spectrogram(fig, ax):
    t=np.linspace(0,2,200); f=np.linspace(0,100,100)
    T,F=np.meshgrid(t,f)
    S=np.exp(-((F-30)**2)/200)*np.exp(-((T-1)**2)/0.5)+np.random.normal(0,0.1,T.shape)
    c=ax.pcolormesh(T,F,S,cmap='hot',shading='auto')
    plt.colorbar(c,ax=ax,label='Power (dB)')
    ax.set_xlabel('Time (s)'); ax.set_ylabel('Frequency (Hz)')
    return [{"type":"spectrogram","peak_freq":30,"peak_time":1.0}]


def generate_connectivity_matrix(fig, ax):
    n=10; labels=[f"ROI{i}" for i in range(n)]
    C=np.random.uniform(-1,1,(n,n)); C=(C+C.T)/2; np.fill_diagonal(C,1)
    im=ax.imshow(C,cmap='RdBu_r',vmin=-1,vmax=1)
    plt.colorbar(im,ax=ax); ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(labels,rotation=45,fontsize=6); ax.set_yticklabels(labels,fontsize=6)
    return [{"type":"connectivity","n_regions":n}]


def generate_psychometric_curve(fig, ax):
    x=np.linspace(-3,3,20); thresh=random.uniform(-0.5,0.5); slope=random.uniform(1,3)
    p=1/(1+np.exp(-slope*(x-thresh)))+np.random.normal(0,0.03,len(x))
    p=np.clip(p,0,1)
    ax.plot(x,p,'ko',ms=5); fit_x=np.linspace(-3,3,100); ax.plot(fit_x,1/(1+np.exp(-slope*(fit_x-thresh))),'r-')
    ax.axhline(0.5,color='gray',ls='--'); ax.axvline(thresh,color='gray',ls='--')
    ax.set_xlabel('Stimulus Strength'); ax.set_ylabel('P(correct)')
    return [{"type":"psychometric","threshold":round(thresh,2),"slope":round(slope,2)}]


def generate_time_series_anomaly(fig, ax):
    years=np.arange(1900,2025); baseline=0
    anomaly=0.01*(years-1950)+0.5*np.sin(2*np.pi*years/60)+np.random.normal(0,0.2,len(years))
    colors=['blue' if a<0 else 'red' for a in anomaly]
    ax.bar(years,anomaly,color=colors,width=1)
    ax.axhline(0,color='k',lw=1); ax.set_xlabel('Year'); ax.set_ylabel('Temperature Anomaly (°C)')
    return [{"type":"anomaly_ts","trend_per_decade":round(0.01*10,3)}]


def generate_hovmoller(fig, ax):
    lon=np.linspace(0,360,72); t=np.arange(365)
    LON,T=np.meshgrid(lon,t)
    Z=np.sin(2*np.pi*(LON-T*0.5)/360)+np.random.normal(0,0.2,LON.shape)
    c=ax.pcolormesh(LON,T,Z,cmap='RdBu_r',shading='auto')
    plt.colorbar(c,ax=ax,label='OLR Anomaly')
    ax.set_xlabel('Longitude'); ax.set_ylabel('Day of Year')
    return [{"type":"hovmoller","propagation_speed":0.5}]


def generate_vertical_profile(fig, ax):
    z=np.linspace(0,30,50); T=-6.5*z/1000*50+15+np.random.normal(0,1,50)
    ax.plot(T,z,'b-o',ms=3,label='Temperature')
    ax.set_xlabel('Temperature (°C)'); ax.set_ylabel('Altitude (km)')
    ax.invert_yaxis() if random.random()>0.5 else None
    if ax.get_legend_handles_labels()[1]: ax.legend(loc=random.choice(['upper left', 'upper right', 'lower left', 'lower right']))
    return [{"type":"vertical_profile","surface_T":round(float(T[0]),1)}]


def generate_climate_ensemble(fig, ax):
    years=np.arange(2000,2100); gt=[]
    for sc,c in zip(['SSP1-2.6','SSP2-4.5','SSP5-8.5'],['blue','orange','red']):
        rate={'SSP1-2.6':0.01,'SSP2-4.5':0.03,'SSP5-8.5':0.06}[sc]
        mean=rate*(years-2000)+np.random.normal(0,0.1,len(years)).cumsum()*0.05
        ax.plot(years,mean,color=c,label=sc)
        ax.fill_between(years,mean-0.5,mean+0.5,color=c,alpha=0.2)
        gt.append({"type":"ensemble","scenario":sc,"warming_2100":round(float(mean[-1]),1)})
    ax.set_xlabel('Year'); ax.set_ylabel('Temperature Change (°C)')
    if ax.get_legend_handles_labels()[1]: ax.legend(fontsize=7, loc=random.choice(['upper left', 'upper right', 'lower left', 'lower right']))
    return gt


def generate_return_period(fig, ax):
    rp=np.logspace(0,3,30); mag=20+10*np.log(rp)+np.random.normal(0,2,len(rp))
    ax.semilogx(rp,mag,'ko',ms=4); fit=20+10*np.log(np.logspace(0,3,100))
    ax.plot(np.logspace(0,3,100),fit,'r-')
    ax.fill_between(np.logspace(0,3,100),fit-3,fit+3,color='red',alpha=0.15)
    ax.set_xlabel('Return Period (years)'); ax.set_ylabel('Magnitude')
    return [{"type":"return_period","100yr_event":round(float(20+10*np.log(100)),1)}]


def generate_ocean_section(fig, ax):
    dist=np.linspace(0,5000,60); depth=np.linspace(0,5000,50)
    D,Z=np.meshgrid(dist,depth)
    T=20*np.exp(-Z/1000)+5*np.sin(2*np.pi*D/5000)+np.random.normal(0,0.5,D.shape)
    c=ax.pcolormesh(D,Z,T,cmap='coolwarm',shading='auto')
    ax.invert_yaxis(); plt.colorbar(c,ax=ax,label='Temperature (°C)')
    ax.contour(D,Z,T,levels=10,colors='k',linewidths=0.3)
    ax.set_xlabel('Distance (km)'); ax.set_ylabel('Depth (m)')
    return [{"type":"ocean_section","sst_range":round(float(T[0].max()-T[0].min()),1)}]


def generate_vector_field(fig, ax):
    x=np.linspace(-5,5,15); y=np.linspace(-5,5,15); X,Y=np.meshgrid(x,y)
    U=-Y+np.random.normal(0,0.2,X.shape); V=X+np.random.normal(0,0.2,X.shape)
    speed=np.sqrt(U**2+V**2)
    ax.quiver(X,Y,U,V,speed,cmap='viridis',alpha=0.8)
    ax.set_xlabel('x'); ax.set_ylabel('y')
    return [{"type":"vector_field","max_speed":round(float(speed.max()),2)}]


def generate_streamline(fig, ax):
    x=np.linspace(-3,3,40); y=np.linspace(-3,3,40); X,Y=np.meshgrid(x,y)
    U=-Y; V=X
    speed=np.sqrt(U**2+V**2)
    ax.streamplot(X,Y,U,V,color=speed,cmap='plasma',density=1.5)
    ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_aspect('equal')
    return [{"type":"streamline"}]


def generate_lift_drag_polar(fig, ax):
    alpha=np.linspace(-5,20,30)
    for name,clmax,c in zip(['NACA0012','NACA2412'],[1.2,1.5],['b','r']):
        cl=0.1*alpha*(1-0.01*alpha**2)+np.random.normal(0,0.02,len(alpha))
        cl=np.clip(cl,-0.5,clmax)
        ax.plot(alpha,cl,f'{c}o-',ms=3,label=name)
    ax.axhline(0,color='k',lw=0.5); ax.axvline(0,color='k',lw=0.5)
    ax.set_xlabel(r'$\alpha$ (°)'); ax.set_ylabel(r'$C_L$')
    if ax.get_legend_handles_labels()[1]: ax.legend(loc=random.choice(['upper left', 'upper right', 'lower left', 'lower right']))
    return [{"type":"lift_drag_polar"}]


def generate_pressure_coefficient(fig, ax):
    x=np.linspace(0,1,50)
    cp_upper=-2*np.sqrt(1-x)*np.exp(-3*x)+np.random.normal(0,0.05,50)
    cp_lower=0.5*(1-x)+np.random.normal(0,0.05,50)
    ax.plot(x,cp_upper,'b-o',ms=2,label='Upper'); ax.plot(x,cp_lower,'r-s',ms=2,label='Lower')
    ax.invert_yaxis(); ax.set_xlabel('x/c'); ax.set_ylabel(r'$C_p$')
    if ax.get_legend_handles_labels()[1]: ax.legend(loc=random.choice(['upper left', 'upper right', 'lower left', 'lower right']))
    return [{"type":"cp_distribution","min_cp":round(float(cp_upper.min()),2)}]


def generate_bode_plot(fig, _):
    fig.clf(); ax1,ax2=fig.subplots(2,1,sharex=True)
    f=np.logspace(-1,4,200); wn=100; zeta=0.3
    s=1j*2*np.pi*f; H=wn**2/(s**2+2*zeta*wn*s+wn**2)
    ax1.semilogx(f,20*np.log10(np.abs(H)),'b-')
    ax1.set_ylabel('Magnitude (dB)'); ax1.grid(True,alpha=0.3)
    ax2.semilogx(f,np.degrees(np.angle(H)),'r-')
    ax2.set_ylabel('Phase (°)'); ax2.set_xlabel('Frequency (Hz)'); ax2.grid(True,alpha=0.3)
    return [{"type":"bode","natural_freq":wn,"damping":zeta}]


def generate_nyquist_plot(fig, ax):
    w=np.logspace(-2,3,500); wn=10; zeta=0.5
    s=1j*w; H=wn**2/(s**2+2*zeta*wn*s+wn**2)
    ax.plot(H.real,H.imag,'b-'); ax.plot(H.real,-H.imag,'b--',alpha=0.5)
    ax.plot(-1,0,'rx',ms=10,mew=2); ax.set_xlabel('Real'); ax.set_ylabel('Imaginary')
    ax.set_aspect('equal'); ax.grid(True,alpha=0.3)
    return [{"type":"nyquist","wn":wn,"zeta":zeta}]


def generate_convergence_plot(fig, ax):
    iters=np.arange(1,101)
    for name,rate,c in zip(['Solver A','Solver B','Solver C'],[0.05,0.08,0.03],['r','b','g']):
        res=10*np.exp(-rate*iters)+np.random.normal(0,0.01,100); res=np.clip(res,1e-6,None)
        ax.semilogy(iters,res,color=c,label=name)
    ax.set_xlabel('Iteration'); ax.set_ylabel('Residual')
    if ax.get_legend_handles_labels()[1]: ax.legend(loc=random.choice(['upper left', 'upper right', 'lower left', 'lower right']))
    return [{"type":"convergence"}]


def generate_pareto_frontier(fig, ax):
    n=100; f1=np.random.uniform(0,10,n); f2=10-f1+np.random.normal(0,2,n)
    ax.scatter(f1,f2,c='gray',s=15,alpha=0.5,label='Feasible')
    idx=np.argsort(f1); pareto_f1=[f1[idx[0]]]; pareto_f2=[f2[idx[0]]]
    for i in idx:
        if f2[i]>=pareto_f2[-1]: pareto_f1.append(f1[i]); pareto_f2.append(f2[i])
    ax.plot(pareto_f1,pareto_f2,'r-o',ms=4,label='Pareto Front')
    ax.set_xlabel('Objective 1'); ax.set_ylabel('Objective 2')
    if ax.get_legend_handles_labels()[1]: ax.legend(loc=random.choice(['upper left', 'upper right', 'lower left', 'lower right']))
    return [{"type":"pareto","n_pareto":len(pareto_f1)}]


def generate_training_curve(fig, ax):
    epochs=np.arange(1,101)
    train_loss=5*np.exp(-0.05*epochs)+0.1+np.random.normal(0,0.05,100)
    val_loss=5*np.exp(-0.04*epochs)+0.3+np.random.normal(0,0.08,100)
    ax.plot(epochs,train_loss,'b-',label='Train'); ax.plot(epochs,val_loss,'r-',label='Val')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Loss'); ax.set_yscale('log')
    if ax.get_legend_handles_labels()[1]: ax.legend(loc=random.choice(['upper left', 'upper right', 'lower left', 'lower right']))
    return [{"type":"training_curve","final_train":round(float(train_loss[-1]),3),"final_val":round(float(val_loss[-1]),3)}]


def generate_scaling_law(fig, ax):
    params=np.logspace(6,10,20); loss=10*params**-0.076+np.random.normal(0,0.01,20)
    ax.loglog(params,loss,'ko',ms=4); fit_x=np.logspace(6,10,100)
    ax.plot(fit_x,10*fit_x**-0.076,'r-',label=r'$L \propto N^{-0.076}$')
    ax.set_xlabel('Parameters'); ax.set_ylabel('Loss')
    if ax.get_legend_handles_labels()[1]: ax.legend(loc=random.choice(['upper left', 'upper right', 'lower left', 'lower right']))
    return [{"type":"scaling_law","exponent":-0.076}]


def generate_confusion_matrix(fig, ax):
    n=random.randint(3,6); labels=[f"C{i}" for i in range(n)]
    cm=np.random.randint(0,100,(n,n)); np.fill_diagonal(cm,np.random.randint(200,500,n))
    im=ax.imshow(cm,cmap='Blues')
    for i in range(n):
        for j in range(n):
            ax.text(j,i,str(cm[i,j]),ha='center',va='center',fontsize=8)
    plt.colorbar(im,ax=ax); ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(labels); ax.set_yticklabels(labels)
    ax.set_xlabel('Predicted'); ax.set_ylabel('True')
    return [{"type":"confusion_matrix","accuracy":round(float(np.trace(cm)/cm.sum()),3)}]


def generate_ablation_plot(fig, ax):
    variants=['Full','No Aug','No Pretrain','No Dropout','Baseline']
    scores=[92.3,89.1,85.7,90.5,78.2]; errs=[0.5,0.8,1.2,0.6,1.5]
    scores=[s+random.uniform(-2,2) for s in scores]; y=range(len(variants))
    ax.barh(y,scores,xerr=errs,color=['green']+['steelblue']*3+['gray'],capsize=3)
    ax.set_yticks(y); ax.set_yticklabels(variants); ax.set_xlabel('Accuracy (%)')
    return [{"type":"ablation","variant":v,"score":round(s,1)} for v,s in zip(variants,scores)]


def generate_calibration_reliability(fig, ax):
    bins=np.linspace(0,1,11); bin_centers=(bins[:-1]+bins[1:])/2
    acc=bin_centers+np.random.normal(0,0.05,10); acc=np.clip(acc,0,1)
    ax.bar(bin_centers,acc,width=0.08,alpha=0.7,color='steelblue',label='Model')
    ax.plot([0,1],[0,1],'r--',label='Perfect')
    ax.set_xlabel('Predicted Probability'); ax.set_ylabel('Observed Frequency')
    if ax.get_legend_handles_labels()[1]: ax.legend(loc=random.choice(['upper left', 'upper right', 'lower left', 'lower right']))
    return [{"type":"calibration_reliability","ece":round(float(np.mean(np.abs(acc-bin_centers))),3)}]


def generate_attention_heatmap(fig, ax):
    tokens_x=['The','cat','sat','on','the','mat','.']; tokens_y=tokens_x[:]; n=len(tokens_x)
    attn=np.random.dirichlet(np.ones(n),n)
    im=ax.imshow(attn,cmap='Purples'); ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(tokens_x,rotation=45); ax.set_yticklabels(tokens_y)
    plt.colorbar(im,ax=ax,label='Attention Weight')
    return [{"type":"attention","n_tokens":n}]


def generate_function_plot(fig, ax):
    x=np.linspace(-5,5,300)
    for label,y,c in [('sin',np.sin(x),'b'),('cos',np.cos(x),'r'),(r'$e^{-x^2}$',np.exp(-x**2),'g')]:
        ax.plot(x,y,color=c,label=label)
    ax.axhline(0,color='k',lw=0.5); ax.axvline(0,color='k',lw=0.5)
    ax.set_xlabel('x'); ax.set_ylabel('f(x)')
    if ax.get_legend_handles_labels()[1]: ax.legend(loc=random.choice(['upper left', 'upper right', 'lower left', 'lower right']))
    return [{"type":"function_plot"}]


def generate_phase_portrait(fig, ax):
    x=np.linspace(-3,3,15); y=np.linspace(-3,3,15); X,Y=np.meshgrid(x,y)
    a=random.uniform(-1,1); b=random.uniform(-1,1)
    U=a*X-Y; V=X+b*Y
    ax.streamplot(X,Y,U,V,color=np.sqrt(U**2+V**2),cmap='autumn',density=1.5)
    ax.plot(0,0,'ko',ms=8)
    ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_aspect('equal')
    return [{"type":"phase_portrait","a":round(a,2),"b":round(b,2)}]


def generate_bifurcation(fig, ax):
    r=np.linspace(2.5,4.0,500)
    for ri in r:
        x=0.5
        for _ in range(200): x=ri*x*(1-x)
        xs=[x]; 
        for _ in range(100): x=ri*x*(1-x); xs.append(x)
        ax.plot([ri]*len(xs),xs,'k,',alpha=0.3)
    ax.set_xlabel('r'); ax.set_ylabel('x*')
    return [{"type":"bifurcation","r_range":[2.5,4.0]}]


def generate_qq_plot(fig, ax):
    data=np.sort(np.random.normal(0,1,100)+np.random.exponential(0.3,100))
    theoretical=np.sort(np.random.normal(0,1,100))
    ax.scatter(theoretical,data,c='k',s=10)
    lims=[min(theoretical.min(),data.min()),max(theoretical.max(),data.max())]
    ax.plot(lims,lims,'r--'); ax.set_xlabel('Theoretical Quantiles'); ax.set_ylabel('Sample Quantiles')
    return [{"type":"qq_plot"}]


def generate_residual_plot(fig, ax):
    fitted=np.random.uniform(0,100,80); residuals=np.random.normal(0,5,80)
    ax.scatter(fitted,residuals,c='k',s=10,alpha=0.6)
    ax.axhline(0,color='r',ls='--')
    ax.set_xlabel('Fitted Values'); ax.set_ylabel('Residuals')
    return [{"type":"residual_plot","mean_resid":round(float(np.mean(residuals)),2)}]


def generate_autocorrelation(fig, ax):
    lags=np.arange(0,30); acf=np.exp(-lags/5)*np.cos(lags/3)+np.random.normal(0,0.05,30)
    acf[0]=1.0
    ax.stem(lags,acf,linefmt='b-',markerfmt='bo',basefmt='k-')
    ax.axhline(1.96/np.sqrt(100),color='r',ls='--'); ax.axhline(-1.96/np.sqrt(100),color='r',ls='--')
    ax.set_xlabel('Lag'); ax.set_ylabel('ACF')
    return [{"type":"autocorrelation"}]


def generate_trace_plot(fig, ax):
    n_iter=1000
    for i,c in enumerate(['blue','red','green']):
        chain=np.cumsum(np.random.normal(0,0.1,n_iter))+random.uniform(-2,2)
        ax.plot(range(n_iter),chain,color=c,alpha=0.7,lw=0.5,label=f'Chain {i+1}')
    ax.set_xlabel('Iteration'); ax.set_ylabel('Parameter Value')
    if ax.get_legend_handles_labels()[1]: ax.legend(fontsize=7, loc=random.choice(['upper left', 'upper right', 'lower left', 'lower right']))
    return [{"type":"trace_plot","n_chains":3}]


def generate_coefficient_plot(fig, ax):
    n=random.randint(5,10); names=[f"Var{i}" for i in range(n)]
    coefs=np.random.normal(0,1,n); ci=np.random.uniform(0.2,0.8,n)
    y=np.arange(n)
    ax.errorbar(coefs,y,xerr=ci,fmt='ko',capsize=3)
    ax.axvline(0,color='r',ls='--')
    ax.set_yticks(y); ax.set_yticklabels(names,fontsize=7); ax.set_xlabel('Coefficient (95% CI)')
    return [{"type":"coefficient","name":n,"coef":round(float(c),3)} for n,c in zip(names,coefs)]


def generate_event_study(fig, ax):
    t=np.arange(-5,6); effects=np.zeros(11); effects[5:]=np.random.uniform(0.5,3,6)
    effects+=np.random.normal(0,0.3,11); ci=np.random.uniform(0.3,0.8,11)
    ax.errorbar(t,effects,yerr=ci,fmt='ko-',capsize=3)
    ax.axvline(0,color='r',ls='--',label='Treatment')
    ax.axhline(0,color='k',ls='-',lw=0.5)
    ax.set_xlabel('Time Relative to Event'); ax.set_ylabel('Effect Size')
    if ax.get_legend_handles_labels()[1]: ax.legend(loc=random.choice(['upper left', 'upper right', 'lower left', 'lower right']))
    return [{"type":"event_study","post_effect":round(float(np.mean(effects[5:])),2)}]


def generate_lorenz_curve(fig, ax):
    pop=np.linspace(0,1,100)
    gini=random.uniform(0.25,0.55)
    income=pop**(1/(1-gini+0.01))
    ax.plot(pop,income,'b-',lw=2,label=f'Lorenz (Gini={gini:.2f})')
    ax.plot([0,1],[0,1],'k--',label='Equality')
    ax.fill_between(pop,pop,income,alpha=0.2)
    ax.set_xlabel('Cumulative Population'); ax.set_ylabel('Cumulative Income')
    if ax.get_legend_handles_labels()[1]: ax.legend(loc=random.choice(['upper left', 'upper right', 'lower left', 'lower right']))
    return [{"type":"lorenz","gini":round(gini,3)}]


def generate_survey_stacked_bar(fig, ax):
    cats=['Q1','Q2','Q3','Q4','Q5']; responses=['Str. Disagree','Disagree','Neutral','Agree','Str. Agree']
    colors=['#d73027','#fc8d59','#ffffbf','#91bfdb','#4575b4']
    y=np.arange(len(cats)); gt=[]
    for i,resp in enumerate(responses):
        vals=[random.uniform(5,30) for _ in cats]
        left=np.zeros(len(cats)) if i==0 else left+prev
        ax.barh(y,vals,left=left,color=colors[i],label=resp)
        prev=np.array(vals)
    ax.set_yticks(y); ax.set_yticklabels(cats); ax.legend(fontsize=6,loc='lower right')
    return [{"type":"survey_stacked","n_questions":len(cats)}]


def generate_economic_timeseries(fig, ax):
    years=np.arange(1990,2025)
    for name,c in zip(['US','EU','China'],['blue','green','red']):
        gdp=100+np.cumsum(np.random.normal(2,1,len(years)))
        ax.plot(years,gdp,color=c,label=name,lw=1.5)
    # recession bands
    for start in [2001,2008,2020]:
        if start<2025: ax.axvspan(start,start+1,color='gray',alpha=0.2)
    ax.set_xlabel('Year'); ax.set_ylabel('GDP Index')
    if ax.get_legend_handles_labels()[1]: ax.legend(loc=random.choice(['upper left', 'upper right', 'lower left', 'lower right']))
    return [{"type":"economic_ts","n_series":3}]

def generate_bubble_chart(fig, ax):
    n = random.randint(15, 50)
    x = np.random.uniform(0, 100, n); y = np.random.uniform(0, 100, n)
    sizes = np.random.uniform(20, 500, n)
    colors = np.random.uniform(0, 1, n)
    sc = ax.scatter(x, y, s=sizes, c=colors, cmap='viridis', alpha=0.6, edgecolors='k')
    plt.colorbar(sc, ax=ax, label='Category Score')
    ax.set_xlabel('GDP per capita'); ax.set_ylabel('Life Expectancy')
    random_style(ax)
    return [{"type": "bubble", "x": round(float(xi),1), "y": round(float(yi),1), "size": round(float(si),1)} for xi,yi,si in zip(x[:5],y[:5],sizes[:5])]

def generate_radar_chart(fig, _):
    fig.clf(); ax = fig.add_subplot(111, polar=True)
    cats = ['Strength', 'Speed', 'Agility', 'Endurance', 'Power', 'Flexibility']
    n = len(cats); angles = np.linspace(0, 2*np.pi, n, endpoint=False).tolist(); angles += angles[:1]
    gt = []
    for name, c in zip(['Athlete A', 'Athlete B'], ['blue', 'red']):
        vals = [random.uniform(3, 10) for _ in cats]; vals += vals[:1]
        ax.plot(angles, vals, color=c, lw=2, label=name)
        ax.fill(angles, vals, color=c, alpha=0.15)
        gt.append({"type": "radar", "name": name, "values": [round(v,1) for v in vals[:-1]]})
    ax.set_xticks(angles[:-1]); ax.set_xticklabels(cats, fontsize=8)
    ax.legend(loc='upper right', fontsize=7)
    return gt

def generate_sankey_flow(fig, ax):
    # Simplified alluvial/flow using stacked horizontal bars
    stages = ['Leads', 'Qualified', 'Proposal', 'Won']
    values = [1000, 600, 300, 120]
    colors = ['#3182bd', '#6baed6', '#9ecae1', '#c6dbef']
    y = np.arange(len(stages))
    ax.barh(y, values, color=colors, edgecolor='k')
    for i, (s, v) in enumerate(zip(stages, values)):
        ax.text(v/2, i, f'{s}\n{v}', ha='center', va='center', fontweight='bold')
    ax.set_yticks([]); ax.set_xlabel('Count')
    ax.set_title('Conversion Funnel')
    return [{"type": "sankey", "stage": s, "value": v} for s, v in zip(stages, values)]

def generate_treemap(fig, ax):
    from matplotlib.patches import Rectangle
    n = random.randint(6, 12)
    values = sorted([random.uniform(10, 100) for _ in range(n)], reverse=True)
    labels = [f"Cat {i}" for i in range(n)]
    colors = plt.cm.Set3(np.linspace(0, 1, n))
    total = sum(values); x, y, w = 0, 0, 1
    h = 1; ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    for i, (v, l) in enumerate(zip(values, labels)):
        frac = v / total
        if i % 2 == 0:
            rect_w = frac * w / (sum(values[i:]) / total) if sum(values[i:]) > 0 else 0.1
            rect = Rectangle((x, y), rect_w, h, facecolor=colors[i], edgecolor='k')
            ax.add_patch(rect)
            ax.text(x + rect_w/2, y + h/2, f'{l}\n{v:.0f}', ha='center', va='center', fontsize=7)
            x += rect_w
        else:
            rect_h = frac * h / (sum(values[i:]) / total) if sum(values[i:]) > 0 else 0.1
            rect = Rectangle((x-rect_w if i>0 else x, y), rect_w if i>0 else w-x, rect_h, facecolor=colors[i], edgecolor='k')
            ax.add_patch(rect); y += rect_h
    ax.set_xticks([]); ax.set_yticks([])
    return [{"type": "treemap", "label": l, "value": round(v,1)} for l, v in zip(labels, values)]

def generate_polar_plot(fig, _):
    fig.clf(); ax = fig.add_subplot(111, polar=True)
    theta = np.linspace(0, 2*np.pi, 100)
    for i in range(random.randint(1, 3)):
        r = 1 + 0.5*np.cos((i+2)*theta) + np.random.normal(0, 0.05, len(theta))
        ax.plot(theta, r, label=f'Mode {i+1}')
    if ax.get_legend_handles_labels()[1]: ax.legend(fontsize=7, loc=random.choice(['upper left', 'upper right', 'lower left', 'lower right']))
    return [{"type": "polar_plot"}]

def generate_area_chart(fig, ax):
    x = np.arange(50)
    n_series = random.randint(2, 4)
    colors = ['#3182bd', '#e6550d', '#31a354', '#756bb1']
    ys = [np.random.uniform(5, 20, 50).cumsum() * 0.1 for _ in range(n_series)]
    ax.stackplot(x, *ys, labels=[f'Series {i+1}' for i in range(n_series)], colors=colors[:n_series], alpha=0.7)
    ax.set_xlabel('Time'); ax.set_ylabel('Value'); ax.legend(loc='upper left', fontsize=7)
    return [{"type": "stacked_area", "n_series": n_series}]

def generate_funnel_chart(fig, ax):
    stages = ['Visitors', 'Sign-ups', 'Trial', 'Paid', 'Enterprise']
    values = sorted([random.randint(100, 10000) for _ in range(5)], reverse=True)
    y = np.arange(len(stages))
    colors = plt.cm.Blues(np.linspace(0.3, 0.9, len(stages)))
    ax.barh(y, values, color=colors, edgecolor='k', height=0.7)
    for i, (s, v) in enumerate(zip(stages, values)):
        ax.text(v + max(values)*0.02, i, f'{v}', va='center', fontsize=9)
    ax.set_yticks(y); ax.set_yticklabels(stages); ax.invert_yaxis()
    ax.set_xlabel('Count')
    return [{"type": "funnel", "stage": s, "value": v} for s, v in zip(stages, values)]

def generate_parallel_coordinates(fig, ax):
    n_vars = random.randint(4, 7); n_samples = 30
    var_names = [f"Var{i}" for i in range(n_vars)]
    data = np.random.randn(n_samples, n_vars)
    # normalize
    data = (data - data.min(axis=0)) / (data.max(axis=0) - data.min(axis=0))
    classes = np.random.choice([0, 1, 2], n_samples)
    colors_pc = ['blue', 'red', 'green']
    x = np.arange(n_vars)
    for i in range(n_samples):
        ax.plot(x, data[i], color=colors_pc[classes[i]], alpha=0.3, lw=1)
    ax.set_xticks(x); ax.set_xticklabels(var_names, fontsize=8)
    ax.set_ylabel('Normalized Value')
    # legend
    for ci, c in enumerate(colors_pc):
        ax.plot([], [], color=c, label=f'Class {ci}')
    if ax.get_legend_handles_labels()[1]: ax.legend(fontsize=7, loc=random.choice(['upper left', 'upper right', 'lower left', 'lower right']))
    return [{"type": "parallel_coords", "n_vars": n_vars, "n_classes": 3}]

def generate_hexbin_plot(fig, ax):
    n = 5000
    x = np.random.normal(0, 1, n); y = x * 0.5 + np.random.normal(0, 1, n)
    hb = ax.hexbin(x, y, gridsize=25, cmap='YlOrRd', mincnt=1)
    plt.colorbar(hb, ax=ax, label='Count')
    ax.set_xlabel('X'); ax.set_ylabel('Y')
    return [{"type": "hexbin", "n_points": n, "correlation": round(float(np.corrcoef(x,y)[0,1]),3)}]

def generate_candlestick(fig, ax):
    n = 40
    dates = np.arange(n)
    close = 100 + np.cumsum(np.random.normal(0, 2, n))
    open_p = close + np.random.normal(0, 1, n)
    high = np.maximum(open_p, close) + np.abs(np.random.normal(0, 1.5, n))
    low = np.minimum(open_p, close) - np.abs(np.random.normal(0, 1.5, n))
    for i in range(n):
        color = 'green' if close[i] >= open_p[i] else 'red'
        ax.plot([dates[i], dates[i]], [low[i], high[i]], color='k', lw=0.8)
        ax.bar(dates[i], abs(close[i]-open_p[i]), bottom=min(open_p[i],close[i]), color=color, width=0.6, edgecolor='k', lw=0.5)
    ax.set_xlabel('Trading Day'); ax.set_ylabel('Price ($)')
    return [{"type": "candlestick", "final_close": round(float(close[-1]),2)}]

def generate_ternary_diagram(fig, ax):
    n = 80
    # Random compositions summing to 1
    raw = np.random.dirichlet([1,1,1], n)
    a, b, c = raw[:,0], raw[:,1], raw[:,2]
    # Convert to Cartesian for plotting in regular axes
    x = 0.5 * (2*b + c) / (a + b + c)
    y = (np.sqrt(3)/2) * c / (a + b + c)
    colors = a  # color by component A fraction
    sc = ax.scatter(x, y, c=colors, cmap='coolwarm', s=20, alpha=0.7, edgecolors='k', lw=0.3)
    plt.colorbar(sc, ax=ax, label='Component A')
    # Draw triangle
    tri_x = [0, 1, 0.5, 0]; tri_y = [0, 0, np.sqrt(3)/2, 0]
    ax.plot(tri_x, tri_y, 'k-', lw=2)
    ax.text(0, -0.05, 'A', ha='center', fontsize=12)
    ax.text(1, -0.05, 'B', ha='center', fontsize=12)
    ax.text(0.5, np.sqrt(3)/2+0.03, 'C', ha='center', fontsize=12)
    ax.set_xlim(-0.1, 1.1); ax.set_ylim(-0.1, 1.0)
    ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
    return [{"type": "ternary", "n_points": n}]

def generate_swarm_plot(fig, ax):
    n_groups = random.randint(3, 5)
    gt = []
    for i in range(n_groups):
        data = np.random.normal(random.uniform(20,60), random.uniform(5,12), random.randint(20,50))
        # jitter x
        x = np.ones_like(data) * i + np.random.uniform(-0.2, 0.2, len(data))
        ax.scatter(x, data, s=15, alpha=0.6, color=['#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd'][i])
        gt.append({"type": "swarm", "group": f"Group {i+1}", "mean": round(float(np.mean(data)),2), "n": len(data)})
    ax.set_xticks(range(n_groups)); ax.set_xticklabels([f"Group {i+1}" for i in range(n_groups)])
    ax.set_ylabel('Value')
    return gt

def generate_pair_plot(fig, _):
    fig.clf()
    n_vars = 3; n = 100
    names = ['Var A', 'Var B', 'Var C']
    data = np.random.randn(n, n_vars)
    data[:,1] = data[:,0]*0.7 + np.random.randn(n)*0.5  # correlate A and B
    axes = fig.subplots(n_vars, n_vars)
    for i in range(n_vars):
        for j in range(n_vars):
            ax = axes[i,j]
            if i == j:
                ax.hist(data[:,i], bins=15, color='steelblue', alpha=0.7)
            else:
                ax.scatter(data[:,j], data[:,i], s=5, alpha=0.5, c='steelblue')
            if i < n_vars-1: ax.set_xticks([])
            else: ax.set_xlabel(names[j], fontsize=7)
            if j > 0: ax.set_yticks([])
            else: ax.set_ylabel(names[i], fontsize=7)
    plt.subplots_adjust(wspace=0.05, hspace=0.05)
    return [{"type": "pair_plot", "n_vars": n_vars}]

def generate_step_histogram(fig, ax):
    n_dists = random.randint(1, 3)
    colors = ['blue', 'red', 'green']; gt = []
    for i in range(n_dists):
        mu = random.uniform(20, 80); sigma = random.uniform(5, 15)
        data = np.random.normal(mu, sigma, random.randint(500, 2000))
        counts, edges = np.histogram(data, bins=random.randint(20, 40))
        ax.step(edges[:-1], counts, where='post', color=colors[i], lw=1.5, label=f'Process {i+1}')
        gt.append({"type": "step_hist", "process": f"Process {i+1}", "mean": round(mu,1), "entries": len(data)})
    if random.random() > 0.5: ax.set_yscale('log')
    ax.set_xlabel(random.choice([r'$m_{jj}$ [GeV]', 'Energy [keV]', 'ADC counts']))
    ax.set_ylabel('Events / bin')
    if ax.get_legend_handles_labels()[1]: ax.legend(loc=random.choice(['upper left', 'upper right', 'lower left', 'lower right']))
    return gt

def generate_ecdf(fig, ax):
    n_dists = random.randint(1, 3)
    colors = ['blue', 'red', 'green']; gt = []
    for i in range(n_dists):
        mu = random.uniform(20, 80); sigma = random.uniform(5, 15)
        data = np.sort(np.random.normal(mu, sigma, 200))
        ecdf = np.arange(1, len(data)+1) / len(data)
        ax.step(data, ecdf, color=colors[i], lw=1.5, label=f'Sample {i+1}')
        gt.append({"type": "ecdf", "sample": f"Sample {i+1}", "median": round(float(np.median(data)),1)})
    ax.set_xlabel('Value'); ax.set_ylabel('ECDF')
    if ax.get_legend_handles_labels()[1]: ax.legend(loc=random.choice(['upper left', 'upper right', 'lower left', 'lower right']))
    ax.axhline(0.5, color='gray', ls='--', alpha=0.5)
    return gt

def generate_known_function(fig, ax):
    """Plot an exact mathematical function for analytically verifiable extraction."""
    func_type = random.choice(['exponential', 'power_law', 'gaussian', 'polynomial', 'damped_oscillation'])
    x = np.linspace(random.uniform(0, 2), random.uniform(5, 20), random.randint(30, 80))
    noise_level = random.uniform(0, 0.1)
    
    if func_type == 'exponential':
        A = round(random.uniform(1, 10), 2); tau = round(random.uniform(0.5, 5), 2)
        y = A * np.exp(-x / tau)
        params = {"function": "exponential_decay", "A": A, "tau": tau}
        eq_text = f'$y = {A}\\cdot e^{{-x/{tau}}}$'
    elif func_type == 'power_law':
        A = round(random.uniform(0.5, 5), 2); n = round(random.uniform(0.5, 3), 2)
        y = A * x**n
        params = {"function": "power_law", "A": A, "n": n}
        eq_text = f'$y = {A}\\cdot x^{{{n}}}$'
    elif func_type == 'gaussian':
        A = round(random.uniform(1, 10), 2); mu = round(random.uniform(3, 15), 2)
        sigma = round(random.uniform(0.5, 3), 2)
        y = A * np.exp(-0.5 * ((x - mu) / sigma)**2)
        params = {"function": "gaussian", "A": A, "mu": mu, "sigma": sigma}
        eq_text = f'$y = {A}\\cdot e^{{-(x-{mu})^2/2\\cdot{sigma}^2}}$'
    elif func_type == 'polynomial':
        degree = random.randint(2, 4)
        coeffs = [round(random.uniform(-2, 2), 2) for _ in range(degree + 1)]
        y = sum(c * x**i for i, c in enumerate(coeffs))
        params = {"function": f"polynomial_deg{degree}", "coefficients": coeffs}
        eq_text = f'Polynomial (deg {degree})'
    else:  # damped_oscillation
        A = round(random.uniform(1, 5), 2); gamma = round(random.uniform(0.1, 1), 2)
        omega = round(random.uniform(1, 5), 2)
        y = A * np.exp(-gamma * x) * np.cos(omega * x)
        params = {"function": "damped_oscillation", "A": A, "gamma": gamma, "omega": omega}
        eq_text = f'$y = {A}\\cdot e^{{-{gamma}x}}\\cos({omega}x)$'
    
    y_noisy = y + np.random.normal(0, noise_level * np.abs(y).max(), len(y))
    
    if random.random() > 0.5:
        ax.plot(x, y_noisy, 'ko', ms=3, label='Data')
        ax.plot(x, y, 'r-', lw=1.5, label='Fit')
    else:
        ax.plot(x, y_noisy, 'b-', lw=1.5)
    
    if random.random() > 0.4:
        ax.text(0.05, 0.9, eq_text, transform=ax.transAxes, fontsize=9,
                bbox=dict(facecolor='wheat', alpha=0.5))
    ax.set_xlabel('x'); ax.set_ylabel('f(x)')
    if random.random() > 0.7 and ax.get_legend_handles_labels()[1]: ax.legend(loc=random.choice(['upper left', 'upper right', 'lower left', 'lower right']))
    random_style(ax)
    return [{"type": "known_function", **params, "noise": round(noise_level, 3)}]

def generate_dense_scatter(fig, ax):
    """Extreme: 5000-15000 overplotted points."""
    n = random.randint(5000, 15000)
    x = np.random.normal(0, 1, n); y = 0.5*x + np.random.normal(0, 1, n)
    ax.scatter(x, y, s=1, alpha=0.05, c='k')
    r = round(float(np.corrcoef(x, y)[0, 1]), 3)
    ax.set_xlabel('X'); ax.set_ylabel('Y')
    ax.text(0.05, 0.9, f'r = {r}', transform=ax.transAxes)
    return [{"type": "dense_scatter", "n_points": n, "correlation": r}]

def generate_sparse_plot(fig, ax):
    """Extreme: only 2-5 data points."""
    n = random.randint(2, 5)
    x = np.sort(np.random.uniform(0, 10, n))
    y = np.random.uniform(1, 50, n)
    err = np.random.uniform(1, 10, n)
    ax.errorbar(x, y, yerr=err, fmt='ko', ms=8, capsize=5)
    ax.set_xlabel(random.choice(['Mass (GeV)', 'Energy (eV)', 'Time (s)']))
    ax.set_ylabel(random.choice(['Rate', 'Cross-section', 'Counts']))
    random_style(ax)
    return [{"type": "sparse", "x": [round(float(v), 2) for v in x],
             "y": [round(float(v), 2) for v in y],
             "yerr": [round(float(v), 2) for v in err]}]

def generate_huge_dynamic_range(fig, ax):
    """Extreme: values spanning many orders of magnitude."""
    x = np.logspace(-5, 5, 40)
    y = 1e10 * x**-2.5 + np.random.lognormal(0, 0.3, len(x)) * 1e3
    ax.loglog(x, y, 'ko-', ms=3)
    ax.set_xlabel(r'$p_T$ [GeV]'); ax.set_ylabel('d$N$/d$p_T$ [GeV$^{-1}$]')
    random_style(ax)
    return [{"type": "huge_range", "x_range_decades": 10, "y_range_decades": round(float(np.log10(y.max()/y.min())), 1)}]

def generate_notick_plot(fig, ax):
    """Extreme: plot with NO tick labels — model must infer from gridlines or give up."""
    x = np.linspace(0, 10, 50)
    y = np.sin(x) + np.random.normal(0, 0.1, 50)
    ax.plot(x, y, 'b-', lw=2)
    ax.set_xticklabels([]); ax.set_yticklabels([])
    ax.grid(True, alpha=0.3)
    return [{"type": "no_ticks", "note": "tick_labels_removed"}]

def generate_extreme_aspect(fig, _):
    """Extreme: very wide or very tall aspect ratio."""
    if random.random() > 0.5:
        fig.set_size_inches(14, 3)  # very wide
    else:
        fig.set_size_inches(3, 14)  # very tall
    ax = fig.add_subplot(111)
    x = np.linspace(0, 50, 100); y = np.sin(x)
    ax.plot(x, y, 'r-')
    ax.set_xlabel('x'); ax.set_ylabel('y')
    return [{"type": "extreme_aspect"}]

def generate_rotated_labels(fig, ax):
    """Plot with rotated tick labels (common in real papers)."""
    cats = [f"Category_{chr(65+i)}_{''.join(random.choices('abcdef', k=5))}" for i in range(8)]
    vals = [random.uniform(10, 100) for _ in cats]
    ax.bar(range(len(cats)), vals, color=plt.cm.Set2(np.linspace(0, 1, len(cats))))
    ax.set_xticks(range(len(cats)))
    angle = random.choice([30, 45, 60, 90])
    ax.set_xticklabels(cats, rotation=angle, ha='right', fontsize=7)
    ax.set_ylabel('Value')
    return [{"type": "rotated_labels", "rotation": angle,
             "categories": cats, "values": [round(v, 1) for v in vals]}]

def generate_broken_axis(fig, _):
    """Plot with a broken/discontinuous y-axis."""
    fig.clf()
    ax1 = fig.add_subplot(211); ax2 = fig.add_subplot(212)
    x = np.arange(10); y = [5, 7, 6, 8, 100, 105, 98, 102, 7, 6]
    ax1.bar(x, y, color='steelblue'); ax2.bar(x, y, color='steelblue')
    ax1.set_ylim(90, 110); ax2.set_ylim(0, 15)
    ax1.spines['bottom'].set_visible(False); ax2.spines['top'].set_visible(False)
    ax1.tick_params(bottom=False); ax1.set_xticklabels([])
    # break marks
    d = 0.015
    kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False)
    ax1.plot((-d, +d), (-d, +d), **kwargs); ax1.plot((1-d, 1+d), (-d, +d), **kwargs)
    kwargs.update(transform=ax2.transAxes)
    ax2.plot((-d, +d), (1-d, 1+d), **kwargs); ax2.plot((1-d, 1+d), (1-d, 1+d), **kwargs)
    return [{"type": "broken_axis", "values": y}]

def generate_3d_bar(fig, _):
    """3D perspective bar chart (notoriously hard to read)."""
    fig.clf(); ax = fig.add_subplot(111, projection='3d')
    x = np.arange(5); y = np.arange(4)
    X, Y = np.meshgrid(x, y)
    Z = np.random.uniform(5, 50, X.shape)
    dx = dy = 0.6
    colors = plt.cm.viridis(Z.flatten() / Z.max())
    ax.bar3d(X.flatten(), Y.flatten(), np.zeros_like(Z).flatten(),
             dx, dy, Z.flatten(), color=colors, alpha=0.8)
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Value')
    return [{"type": "3d_bar", "values": Z.tolist()}]


# --- Auxiliary sub-task extractors ---

def extract_axis_info(fig_meta):
    """Extract axis metadata from a figure for the axis-reading sub-task."""
    return {
        "task_type": "axis_info",
        "x_label": fig_meta.get("x_label", ""),
        "y_label": fig_meta.get("y_label", ""),
        "x_scale": fig_meta.get("x_scale", "linear"),
        "y_scale": fig_meta.get("y_scale", "linear"),
        "x_inverted": fig_meta.get("x_inverted", False),
        "y_inverted": fig_meta.get("y_inverted", False),
    }

def extract_element_count(fig_meta):
    """Extract structural element counts for the detection sub-task."""
    return {
        "task_type": "element_count",
        "n_series": fig_meta.get("n_series", 1),
        "has_legend": fig_meta.get("has_legend", False),
        "has_colorbar": fig_meta.get("has_colorbar", False),
        "has_error_bars": fig_meta.get("has_error_bars", False),
        "has_grid": fig_meta.get("has_grid", False),
        "plot_type": fig_meta.get("plot_type", "unknown"),
    }

def capture_fig_meta(fig, ax, plot_type_name):
    """Capture axis and element metadata from a rendered figure."""
    meta = {"plot_type": plot_type_name}
    try:
        meta["x_label"] = ax.get_xlabel() or ""
        meta["y_label"] = ax.get_ylabel() or ""
        meta["x_scale"] = ax.get_xscale()
        meta["y_scale"] = ax.get_yscale()
        meta["x_inverted"] = bool(ax.get_xlim()[0] > ax.get_xlim()[1])
        meta["y_inverted"] = bool(ax.get_ylim()[0] > ax.get_ylim()[1])
        meta["has_legend"] = bool(ax.get_legend() is not None)
        meta["has_grid"] = bool(any(l.get_visible() for l in ax.get_xgridlines()))
        meta["n_series"] = int(len(ax.get_lines()) + len(ax.collections))
        meta["has_error_bars"] = bool(any(isinstance(c, plt.matplotlib.container.ErrorbarContainer) 
                                     for c in ax.containers)) if hasattr(ax, 'containers') else False
        meta["has_colorbar"] = bool(len(fig.axes) > len([a for a in fig.axes if a.get_label() != '<colorbar>']))
    except Exception:
        pass
    return meta


def generate_plot(output_dir, num_samples, degrade_fraction=0.3, aux_tasks=False):
    global plot_types, multi_panel_types
    os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
    metadata_path = os.path.join(output_dir, "metadata.jsonl")
    
    plot_types = [
        generate_scatter,
        generate_fit,
        generate_clustering,
        generate_bar,
        generate_grouped_bar,
        generate_boxplot,
        generate_pie,
        generate_histogram,
        generate_density,
        generate_hep_brazil,
        generate_heatmap,
        generate_contour,
        generate_corner_plot,
        generate_contour_overlay,
        generate_bump_hunt,
        generate_stacked_ratio,
        generate_double_y_axis,
        generate_multi_line_log,
        generate_stacked_histogram,
        generate_residual_bump,
        generate_ashby_chart,
        generate_phase_diagram,
        generate_parity_grid,
        generate_stress_strain,
        generate_volcano_plot,
        generate_roc_curve,
        generate_light_curve,
        generate_line_plot,
        generate_violin_plot,
        generate_spatial_map,
        generate_invariant_mass,
        generate_pt_spectrum,
        generate_pull_plot,
        generate_correlation_matrix,
        generate_unfolded_xsec,
        generate_efficiency_map,
        generate_sky_map,
        generate_sed,
        generate_hr_diagram,
        generate_power_spectrum,
        generate_redshift_distribution,
        generate_mass_radius,
        generate_residual_map,
        generate_band_structure,
        generate_dos,
        generate_xrd,
        generate_raman_spectrum,
        generate_magnetization,
        generate_resistivity,
        generate_nmr_spectrum,
        generate_mass_spectrum_chem,
        generate_uv_vis,
        generate_chromatogram,
        generate_reaction_coordinate,
        generate_kinetic_trace,
        generate_titration_curve,
        generate_calibration_curve,
        generate_spectroscopy_2d,
        generate_clustered_heatmap,
        generate_manhattan_plot,
        generate_survival_curve,
        generate_dose_response,
        generate_flow_cytometry,
        generate_forest_plot,
        generate_epidemic_curve,
        generate_bland_altman,
        generate_waterfall_plot,
        generate_spaghetti_plot,
        generate_raster_plot,
        generate_psth,
        generate_tuning_curve,
        generate_eeg_traces,
        generate_spectrogram,
        generate_connectivity_matrix,
        generate_psychometric_curve,
        generate_time_series_anomaly,
        generate_hovmoller,
        generate_vertical_profile,
        generate_climate_ensemble,
        generate_return_period,
        generate_ocean_section,
        generate_vector_field,
        generate_streamline,
        generate_lift_drag_polar,
        generate_pressure_coefficient,
        generate_bode_plot,
        generate_nyquist_plot,
        generate_convergence_plot,
        generate_pareto_frontier,
        generate_training_curve,
        generate_scaling_law,
        generate_confusion_matrix,
        generate_ablation_plot,
        generate_calibration_reliability,
        generate_attention_heatmap,
        generate_function_plot,
        generate_phase_portrait,
        generate_bifurcation,
        generate_qq_plot,
        generate_residual_plot,
        generate_autocorrelation,
        generate_trace_plot,
        generate_coefficient_plot,
        generate_event_study,
        generate_lorenz_curve,
        generate_survey_stacked_bar,
        generate_economic_timeseries,
        generate_bubble_chart,
        generate_radar_chart,
        generate_sankey_flow,
        generate_treemap,
        generate_polar_plot,
        generate_area_chart,
        generate_funnel_chart,
        generate_parallel_coordinates,
        generate_hexbin_plot,
        generate_candlestick,
        generate_ternary_diagram,
        generate_swarm_plot,
        generate_pair_plot,
        generate_step_histogram,
        generate_ecdf,
        generate_known_function,
        generate_dense_scatter,
        generate_sparse_plot,
        generate_huge_dynamic_range,
        generate_notick_plot,
        generate_extreme_aspect,
        generate_rotated_labels,
        generate_broken_axis,
        generate_3d_bar,
    ]
    
    multi_panel_types = {"bode_plot", "corner_plot", "unfolded_xsec", "invariant_mass", "sky_map", "stacked_ratio", "clustered_heatmap", "parity_grid"}
    
    # Define the worker initialization to pass global variables if needed, 
    # but since it's a Linux fork (usually), globals are inherited. We'll pass explicit args.
    
    tasks = [(i, output_dir, degrade_fraction, aux_tasks) for i in range(num_samples)]
    
    num_cores = multiprocessing.cpu_count()
    print(f"Generating data using {num_cores} CPU cores in parallel...")
    
    chunk_size = 5 # Strict small chunksize so tqdm updates constantly!
    
    with open(metadata_path, 'a') as f:
        with multiprocessing.Pool(processes=num_cores) as pool:
            for result_lines in tqdm(pool.imap_unordered(_generate_single_worker, tasks, chunksize=chunk_size), total=num_samples, desc="Generating Complex Data"):
                for line in result_lines:
                    f.write(line + "\n")

def _generate_single_worker(args):
    i, output_dir, degrade_fraction, aux_tasks = args
    result_lines = []
    
    # Need to set random seed per worker so they don't all generate the same data
    np.random.seed()
    random.seed()
    
    # Apply a random plot style to diversify training data
    style_choice = random.choice([
        'default', 'default', 'default',  # weight default more
        'ggplot', 'bmh', 'classic', 'fivethirtyeight',
        'seaborn-v0_8-darkgrid', 'seaborn-v0_8-whitegrid', 'seaborn-v0_8-ticks',
        'science_default', 'science_ieee', 'science_nature', 'science_scatter',
        'hep_cms', 'hep_atlas', 'hep_alice', 'hep_lhcb'
    ])
    
    plt.style.use('default') # reset first
    if style_choice.startswith('science_'):
        import scienceplots
        substyle = style_choice.split('_')[1]
        if substyle == 'default':
            plt.style.use(['science', 'no-latex'])
        else:
            plt.style.use(['science', substyle, 'no-latex'])
    elif style_choice.startswith('hep_'):
        import mplhep as hep
        exp = style_choice.split('_')[1].upper()
        if exp == 'CMS': hep.style.use(hep.style.CMS)
        elif exp == 'ATLAS': hep.style.use(hep.style.ATLAS)
        elif exp == 'ALICE': hep.style.use(hep.style.ALICE)
        elif exp == 'LHCB': hep.style.use(hep.style.LHCb)
    elif style_choice != 'default':
        plt.style.use(style_choice)
        
    plt.rcParams['legend.loc'] = random.choice(['upper left', 'upper right', 'lower left', 'lower right'])
    
    fig, ax = plt.subplots(figsize=(random.uniform(6.0, 9.0), random.uniform(6.0, 9.0)))
    generator = random.choice(plot_types)
    plot_type_name = generator.__name__.replace('generate_', '')
    
    try:
        data_points = generator(fig, ax)

    except Exception as e:
        plt.close(fig)
        return []
    try:
        main_ax = fig.axes[0] if fig.axes else ax
        fig_meta = capture_fig_meta(fig, main_ax, plot_type_name)
    except Exception:
        fig_meta = {"plot_type": plot_type_name}
    image_dir = os.path.join(output_dir, "images", plot_type_name)
    os.makedirs(image_dir, exist_ok=True)
    image_filename = f"image_{i:05d}.png"
    image_path = os.path.join(image_dir, image_filename)
    plt.savefig(image_path, bbox_inches='tight', dpi=random.randint(80, 150))
    plt.close(fig)
    degradation_applied = []
    if random.random() < degrade_fraction:
        try:
            degradation_applied = degrade_image(image_path)
        except Exception:
            pass
    structured_gt = {
        "figure_type": "multi_panel" if plot_type_name in multi_panel_types else "single_panel",
        "image_quality": "degraded" if degradation_applied else "clean",
        "degradation_effects": degradation_applied,
        "panels": [
            {
                "panel_id": "A",
                "plot_type": plot_type_name,
                "data_series": data_points
            }
        ]
    }
    ground_truth = {"gt_parse": structured_gt, "task": "p2n"}
    metadata_entry = {
        "file_name": f"images/{plot_type_name}/{image_filename}",
        "ground_truth": json.dumps(ground_truth)
    }
    result_lines.append(json.dumps(metadata_entry))
    if aux_tasks:
        ax_info = extract_axis_info(fig_meta)
        if ax_info:
            aux_entry = {
                "file_name": f"images/{plot_type_name}/{image_filename}",
                "ground_truth": json.dumps({"gt_parse": ax_info, "task": "axis_info"})
            }
            result_lines.append(json.dumps(aux_entry))
        elem_info = extract_element_count(fig_meta)
        if elem_info:
            aux_entry = {
                "file_name": f"images/{plot_type_name}/{image_filename}",
                "ground_truth": json.dumps({"gt_parse": elem_info, "task": "element_count"})
            }
            result_lines.append(json.dumps(aux_entry))
    return result_lines

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic scientific plots.")
    parser.add_argument("--output_dir", type=str, default="data", help="Output directory")
    parser.add_argument("--samples", type=int, default=100, help="Number of samples to generate")
    parser.add_argument("--degrade_fraction", type=float, default=0.3,
                        help="Fraction of images to degrade (0.0-1.0). Simulates scans, photocopies, aging.")
    parser.add_argument("--aux_tasks", action="store_true", default=False,
                        help="Emit auxiliary sub-task metadata (axis reading, element counting) for each image.")
    args = parser.parse_args()
    
    generate_plot(args.output_dir, args.samples, args.degrade_fraction, args.aux_tasks)
    suffix = " + aux tasks" if args.aux_tasks else ""
    print(f"Generated {args.samples} samples ({args.degrade_fraction*100:.0f}% degraded{suffix}) in {args.output_dir}")

