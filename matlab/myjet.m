function cmap = myjet(ncolors)
% Customized MATLAB heatmap specific for visualizations similar to what's in 3D slicer
% To deal with the ugly brown matlab shows in jet colormap

cmap = jet(ncolors);
redidx = find(ismember(cmap,[1 0 0],'rows'),1); %index at which red RGB values occur

dx = 0.005;
if redidx < ncolors
    cmap(redidx+1:ncolors,:) = [(1-dx:-dx:1-dx*(ncolors-redidx))' zeros(ncolors-redidx,1) zeros(ncolors-redidx,1)]; %essentially, reduce the drop off rate of the R value
end

end