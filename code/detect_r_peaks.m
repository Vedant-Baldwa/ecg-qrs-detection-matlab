function [r_locs, r_vals, peak_found] = detect_r_peaks(snips, params)

[N, L] = size(snips);
fs = params.fs;
% design bandpass
Wn = [params.bp_low params.bp_high] / (fs/2);
[b,a] = butter(params.bp_order, Wn, 'bandpass');

min_dist = max(1, round(params.min_dist_ms*1e-3*fs));
centerIdx = ceil(L/2);
center_win = max(1, round(params.center_window_ms*1e-3*fs));

r_locs = nan(N,1);
r_vals = nan(N,1);
peak_found = false(N,1);

for i = 1:N
    row = snips(i,:)';
    % bandpass (try filtfilt; fallback to filter)
    try
        xf = filtfilt(b,a,row);
    catch
        xf = filter(b,a,row);
    end
    absxf = abs(xf);
    dyn = max(absxf) - min(absxf);
    prom = params.prom_scale * max(dyn, 1e-6);

    % find peaks on abs bandpassed signal
    [pks, locs, w, p] = findpeaks(absxf, 'MinPeakProminence', prom, 'MinPeakDistance', min_dist);
    % if none found, reduce threshold slightly
    if isempty(locs)
        prom2 = max(0.05 * dyn, 1e-6);
        [pks, locs, w, p] = findpeaks(absxf, 'MinPeakProminence', prom2, 'MinPeakDistance', round(min_dist/2));
    end

    % choose candidate
    chosen = NaN;
    if ~isempty(locs)
        % prefer candidate nearest to center within center window
        diffs = abs(locs - centerIdx);
        in_center = find(diffs <= center_win);
        if ~isempty(in_center)
            % choose most prominent among in_center
            [~, idxMax] = max(p(in_center)); chosen = locs(in_center(idxMax));
        else
            % choose globally most prominent
            [~, idxMax] = max(p); chosen = locs(idxMax);
        end
        peak_found(i) = true;
    else
        % fallback: global max of abs bandpassed
        [~, chosen] = max(absxf);
        peak_found(i) = true;
    end

    chosen = round(chosen);
    chosen = max(1, min(L, chosen));
    r_locs(i) = chosen;
    r_vals(i) = xf(chosen);
end
end
