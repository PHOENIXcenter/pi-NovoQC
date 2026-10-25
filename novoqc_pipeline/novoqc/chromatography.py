import numpy as np
from scipy.signal import savgol_filter
from scipy.stats import pearsonr
from pyteomics.mass import Composition, isotopologues, isotopic_composition_abundance

def extract_3d_points(ms1_spectra, precursor_mz, charge, ppm_tol=10, rt_range=None, max_iso=10):
    points = []
    for i, spec in enumerate(ms1_spectra):
        rt = float(spec['scanList']['scan'][0]['scan start time'])
        if rt_range and not (rt_range[0] <= rt <= rt_range[1]):
            continue
        mzs = spec['m/z array']
        ints = spec['intensity array']
        for iso in range(max_iso+1):
            mz0 = precursor_mz + iso * 1.003355 / charge
            mz_tol = mz0 * ppm_tol * 1e-6
            mask = (mzs >= mz0 - mz_tol) & (mzs <= mz0 + mz_tol)
            for m, inten in zip(mzs[mask], ints[mask]):
                points.append([mz0, rt, inten, iso, m])
    return np.asarray(points, dtype=float).reshape(-1, 5)


def get_xic_curve(pts, rts_grid):
    # Sum intensities at each retention time to obtain an aligned XIC.
    rt_to_intensity = {}
    for pt in pts:
        rt = pt[1]
        inten = pt[2]
        rt_to_intensity[rt] = rt_to_intensity.get(rt, 0) + inten
    xic = np.array([rt_to_intensity.get(rt, 0) for rt in rts_grid])
    return xic


def optimize_rt_window(xic, rts, min_ratio=0.1):
    # Smooth the monoisotopic XIC first.
    window = min(9, len(xic) if len(xic) % 2 else len(xic) - 1)
    xic_smooth = savgol_filter(xic, window_length=window, polyorder=2) if window >= 3 else xic.copy()
    main_idx = np.argmax(xic_smooth)
    # Extend on both sides of the apex until the signal falls below max_value * min_ratio.
    left = main_idx
    right = main_idx
    max_val = xic_smooth[main_idx]
    while left > 0 and xic_smooth[left] > max_val*min_ratio:
        left -= 1
    while right < len(xic_smooth)-1 and xic_smooth[right] > max_val*min_ratio:
        right += 1
    return rts[left], rts[right], main_idx, xic_smooth


def theoretical_isotope_distribution(sequence, n_iso=3):
    """
    Compute the theoretical peptide isotope distribution.
    Return normalized theoretical abundances for the isotope peaks.
    """
    comp = Composition(sequence)
    iso = isotopologues(comp)
    probs = np.zeros(n_iso)
    for i, prob_iso in enumerate(iso):
        if i < n_iso:
            probs[i] = isotopic_composition_abundance(prob_iso)
        else:
            break        
    probs = probs / np.sum(probs)
    return probs


def distribution_correlation(obs, theo):
    obs = obs / (obs.sum()+1e-12)
    pearson = np.corrcoef(obs, theo)[0,1]
    rss = np.sum((obs-theo)**2)
    angle = np.dot(obs, theo)/(np.linalg.norm(obs)*np.linalg.norm(theo)+1e-12)
    return pearson, angle, rss


def process_peptide_feature(ms1_spectra, sequence, precursor_mz, charge, ms2_rt, isolation_window, 
                            ppm_tol=10, max_iso=5, min_frac=0.05, noise_multiplier=2, iso_corr_thresh=0.98, 
                           pearson_thresh=0.95, angle_thresh=0.95, rss_thresh=0.05, snr_thresh=2):
    """
    Estimate chromatographic support using the monoisotopic peak window:
    1. Define the integration window from the monoisotopic XIC.
    2. Integrate each isotope peak within that window.
    3. Retain isotope peaks whose XIC correlation exceeds the threshold.
    """
    # 1. Extract MS1 points across retention times.
    rts_grid = np.array([float(spec['scanList']['scan'][0]['scan start time']) for spec in ms1_spectra])
    points = extract_3d_points(ms1_spectra, precursor_mz, charge, ppm_tol, max_iso=max_iso)
    
    # 2. Define the integration window from the monoisotopic peak (iso_idx == 0).
    mono_pts = points[points[:,3]==0]
    mono_xic = get_xic_curve(mono_pts, rts_grid)
    if mono_xic.sum() == 0:
        return None
    rt_min, rt_max, main_idx, xic_smooth = optimize_rt_window(mono_xic, rts_grid)
    window_mask = (rts_grid >= rt_min) & (rts_grid <= rt_max)
    
    # 3. Extract each isotope XIC, compare it with mono_xic and accumulate its signal.
    iso_areas = []
    iso_corrs = []
    for iso in range(max_iso+1):
        iso_pts = points[points[:,3]==iso]
        xic = get_xic_curve(iso_pts, rts_grid)
        xic_win = xic[window_mask]
        mono_xic_win = mono_xic[window_mask]
        # Require signal in the monoisotopic integration window.
        if xic_win.sum() > 0 and mono_xic_win.sum() > 0:
            # Compute the XIC correlation.
            corr = pearsonr(xic_win, mono_xic_win)[0] if len(xic_win) > 2 else 0
        else:
            corr = 0
        iso_areas.append(xic_win.sum())
        iso_corrs.append(corr)
    iso_areas = np.array(iso_areas)
    iso_corrs = np.array(iso_corrs)    

    # 4. Identify supported isotope peaks.
    main_area = iso_areas[0]
    # Estimate the noise baseline.
    all_ints = points[:,2]
    noise = np.percentile(all_ints, 10) if len(all_ints) > 0 else 0
    min_area = max(main_area * min_frac, noise * noise_multiplier)
    valid_mask = (iso_areas > min_area) & (iso_corrs > iso_corr_thresh)
    num_valid = valid_mask.sum()

    # 5. Use supported isotope peaks for distribution agreement, weighted m/z and total intensity.
    total_intensity = iso_areas[valid_mask].sum()
    valid_isos = np.where(valid_mask)[0]
    mz_numerator = 0 # Intensity-weighted m/z numerator.
    mz_denominator = 0 # Total intensity of supported isotope peaks.
    for iso in valid_isos:
        iso_pts = points[points[:,3]==iso]
        mz_numerator += np.sum(iso_pts[:,2] * iso_pts[:,4])
        mz_denominator += iso_pts[:,2].sum()
    mz_corrected = mz_numerator / (mz_denominator + 1e-8) if mz_denominator > 0 else np.nan

    # 6. Record the main-peak retention-time interval.
    rt_peak = rts_grid[main_idx]
    width = rt_max - rt_min

    # 7. Compare with the theoretical isotope distribution.
    theo_isotopes = theoretical_isotope_distribution(sequence, n_iso=len(iso_areas))
    pearson, angle, rss = distribution_correlation(iso_areas, theo_isotopes)
    
    # SNR
    signal = np.percentile(mono_xic[window_mask], 90)
    noise = np.percentile(mono_xic[window_mask], 10)
    snr = signal/(noise+1e-8)
    
    # Assign a chromatographic-support category.
    passes = {
        'pearson': pearson > pearson_thresh,
        'angle': angle > angle_thresh,
        'rss': rss < rss_thresh,
        'snr': snr > snr_thresh,
    }
    grade = 'A' if all(passes.values()) else (
        'B' if (pearson > 0.85 and angle > 0.85 and rss < 0.15 and snr > 1) else 'C'
    )
    num_scans = int(window_mask.sum())
    
    features = {
        'Number of scans': num_scans, 
        'area': total_intensity,
        'iso_area':iso_areas[valid_mask],
        'theo_isotopes': theo_isotopes.tolist(),
        'mz_raw': precursor_mz,
        'mz_corr': mz_corrected,
        'rt_min': rt_min,
        'rt_max': rt_max,
        'rt_peak': rt_peak,
        'rt_width': width,
        'num_isotopes': num_valid,
        'XIC_pearson': iso_corrs[valid_mask],
        'pearson': pearson,
        'spec_angle': angle,
        'rss': rss,
        'snr': snr,
        'grade': grade,
        'qc_detail': passes,
        'is_trustworthy': (grade == 'A')
        }
        
    return features


def digestion_settings():
    return [("Trypsin", 'C'),
            ("AspN", 'N'),
            ("LysN", 'N'),
            ("LysC", 'C'),
            ("Chymotrypsin", 'C'),
            ("GluC", 'C')]


def general_enzyme_digestion_with_separator(protein_sequence, enzyme, max_missed_cleavages=2, cut_at='C'):
    """
    Perform general enzyme digestion on a protein sequence based on the specified enzyme and cut direction (N/C).
    Allows for a specified number of missed cleavages.
    Includes a separator '|' in the protein sequence at the cleavage sites.
    Does not consider the special case of P following Trypsin and Chymotrypsin cleavage sites.
    """
    
    if enzyme == 'Trypsin':
        digestion_sites = ['K', 'R']  # Trypsin cuts after K or R
    elif enzyme == 'AspN':
        digestion_sites = ['D']  # Asp-N cuts before D
    elif enzyme == 'LysN':
        digestion_sites = ['K']  # Lys-N cuts before K
    elif enzyme == 'LysC':
        digestion_sites = ['K']  # Lys-C cuts after K
    elif enzyme == 'Chymotrypsin':
        digestion_sites = ['F', 'W', 'Y']  # Chymotrypsin cuts after F, W, or Y
    elif enzyme == 'GluC':
        digestion_sites = ['E']  # Glu-C cuts after E or D
    else:
        raise ValueError("Unsupported enzyme. Please use 'Trypsin', 'AspN', 'LysN', 'LysC', 'Chymotrypsin', or 'GluC'.")

    # Adding separators to the sequence
    separated_sequence = ""
    for i in range(len(protein_sequence)):
        separated_sequence += protein_sequence[i]
        if i < len(protein_sequence) - 1:
            if enzyme in ['Trypsin', 'LysC', 'Chymotrypsin', 'GluC'] and protein_sequence[i] in digestion_sites:
                separated_sequence += ' | '
            elif (enzyme == 'AspN' or enzyme == 'LysN') and protein_sequence[i+1] in digestion_sites:
                separated_sequence += ' | '

    return separated_sequence


def generate_peptide_combinations(fragments, min_length=6, max_length=40):
    # Collect combinations that meet the criteria.
    valid_combinations = []
    n = len(fragments)

    # Generate contiguous subsequence combinations.
    for i in range(n):
        for j in range(i + 1, n + 1):
            # Extract the subsequence from position i to position j.
            peptide = ''.join(fragments[i:j])
            if min_length <= len(peptide) <= max_length:
                valid_combinations.append(peptide)
    
    return valid_combinations

