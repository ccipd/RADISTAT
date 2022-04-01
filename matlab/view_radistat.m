function view_radistat(img,mask,feature_map,supervoxel_labels,cluster_values,expression_values,texture_vec,spatial_vec,cmin,cmax)
% Visualize RADISTAT expression maps based on output from RADISTAT function

% Check for necessarily files in path
if exist('display_feature_map.m','file')~=2
    error('Get display_feature_map.m into your path.');
end

if exist('myjet.m','file')~=2
    error('Get myjet.m into your path.');
end

% Check inputs
if ~all(size(img)==size(mask))
    error('IMG and MASK must be same size.');
end

if ~all(size(mask)==size(feature_map))
    error('MASK and FEATURE_MAP must be same size.');
end
    
if isempty(find(mask>0))
   error('No nonzero MASK values. Cannot visualize RADISTAT.');
   return;
end

featvec = feature_map(mask>0);
if (length(find(mask>0))~=length(featvec))
    error('check MASK or FEATURE_VALUES -- not the same number of pixels');
end

if nargin<10 || isempty(cmin) || isempty(cmax)
    cmin = min(featvec(:));
    cmax = max(featvec(:));
end

%% Moving on        
fprintf('viewing.\n');
 
nbins = 3;
edges = 0:1/nbins:1;  
 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if size(img,3)==1
 
ax1=subplot(2,4,1);
imagesc(img); title('MRI slice'); 
c = colorbar('east');c.Color = 'w'; c.FontSize = 12;
caxis([min(img(:)) max(img(:))]); 
cmap1 = gray(256); colormap(ax1,cmap1);colorbar off;
axis image; axis off; %axis(a);
 
ax2=subplot(2,4,2);
%**********************************************%
% display_feature_map(img,mask,rescale_range(featvec,all_ints(60),all_ints(97))); %help with making featvec look similar to radistat
display_feature_map(img,mask,featvec);
%**********************************************%
title('Original Feature ints');
caxis([cmin cmax]);
cmap2=myjet(256);cmap2(1,:)=[0 0 0];
colormap(ax2,cmap2); colorbar off; %axis(a);
 
ax3=subplot(2,4,3);
display_feature_map(img,mask,supervoxel_labels); title('supervoxel clusters');
caxis([0 length(unique(supervoxel_labels))]);
cmap3=colorcube(length(unique(supervoxel_labels))+1);cmap3(1,:)=[0 0 0];colormap(ax3,cmap3)
%axis(a);
 
ax4=subplot(2,4,4);
display_feature_map(img,mask,cluster_values);title('Clustered ints');
caxis([cmin cmax]);
cmap4=myjet(length(unique(cluster_values)));cmap4(1,:)=[0 0 0];
redidx = find(ismember(cmap4,[1 0 0],'rows'),1); cmap4(redidx:size(cmap4,1),:) = repmat([1 0 0],length(redidx:size(cmap4,1)),1);
colormap(ax4,cmap4);
%axis(a);
 
ax5=subplot(2,4,5);
display_feature_map(img,mask,expression_values);title('RADISTAT map');
caxis([0 1]);
cmap5=[0 0 0; 0 1 1; 1 1 0; 1 0 0];colormap(ax5,cmap5);
set(ax5.Colorbar,'Limits',[0.28 1],'Ticks',[0.4,.62,.88],'TickLabels',[]);%'TickLabels',{'Low','Medium','High'}');
yh = ylabel(ax5.Colorbar,'Low       Medium       High'); set(yh,'Rotation',90);
colorbar off;%axis(a);
 
subplot(2,4,6);
cmap = jet(length(edges));
for i = 1:length(edges)-1
   h = bar(i,texture_vec(i),'facecolor',cmap(i+1,:),'edgecolor','k');set(h,'BarWidth',1);hold on;
end
set(gca,'Ylim',[0 1],'box','off','Xtick',1:length(edges)-1,'Fontname','Timesnewroman','Fontsize',12,'XtickLabel',{'L','M','H'},'Linewidth',2);%'XtickLabel',{'Low','Medium','High'});
title('Textural');
ylabel('Relative Distribution of Expression Levels');
xlabel('Feature Expression Levels');
 
subplot(2,4,7);
h = bar(spatial_vec,'facecolor','m','edgecolor','k');set(h,'BarWidth',1);
set(gca,'Ylim',[0 1],'box','off','Fontname','Timesnewroman','Fontsize',12,'Linewidth',2,'XTickLabel',{'L<->M','L<->H','M<->H'},'Fontsize',12);
title('Spatial');
ylabel('Relative Frequency','Fontsize',14);
xlabel('Feature Expression Edge Pairs','Fontsize',14);
 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
elseif size(img,3)>1
  
begin_index = find(mask>0,1,'first');
end_index = find(mask>0,1,'last');
 
[~,~,begin_slice] = ind2sub(size(mask),begin_index);
[~,~,end_slice] = ind2sub(size(mask),end_index);
slice = round((end_slice+begin_slice)/2);
 
featvol = createFeatVol(featvec,mask);
supervol = createFeatVol(supervoxel_labels,mask);
clustvol = createFeatVol(cluster_values,mask);
radvol = createFeatVol(expression_values,mask);
 
imgslice = img(:,:,slice);maskslice = mask(:,:,slice);featslice = featvol(:,:,slice);
superslice = supervol(:,:,slice);clustslice = clustvol(:,:,slice);radslice = radvol(:,:,slice);
 
ax1=subplot(2,4,1);
imagesc(imgslice); title('MRI slice'); 
c = colorbar('east');c.Color = 'w'; c.FontSize = 12;
caxis([min(imgslice(:)) max(imgslice(:))]); 
cmap1 = gray(256); colormap(ax1,cmap1);colorbar off;
axis image; axis off; 
 
ax2=subplot(2,4,2);
%**********************************************%
% display_feature_map(imgslice,maskslice,rescale_range(featslice(maskslice==1),all_ints(60),all_ints(90))); %help with making featvec look similar to radistat
featvec = featslice(maskslice>0);
display_feature_map(imgslice,maskslice,featvec); 

%**********************************************%
 
title('Original Feature ints');
caxis([cmin cmax]);
cmap2=myjet(256);cmap2(1,:)=[0 0 0];
colormap(ax2,cmap2); %colorbar off;
 
ax3=subplot(2,4,3);
display_feature_map(imgslice,maskslice,superslice(maskslice==1)); title('supervoxel clusters');
caxis([0 length(unique(cluster_values))]);
cmap3=colorcube(length(unique(cluster_values))+1);cmap3(1,:)=[0 0 0];colormap(ax3,cmap3)
 
ax4=subplot(2,4,4);
display_feature_map(imgslice,maskslice,clustslice(maskslice==1));title('Clustered ints');
caxis([cmin cmax]);
cmap4=myjet(length(unique(cluster_values)));cmap4(1,:)=[0 0 0];
colormap(ax4,cmap4);
 
ax5=subplot(2,4,5);
display_feature_map(imgslice,maskslice,radslice(maskslice==1));title('RADISTAT map');
caxis([0 1]);
cmap5=[0 0 0; 0 1 1; 1 1 0; 1 0 0];colormap(ax5,cmap5);
set(ax5.Colorbar,'Limits',[0.28 1],'Ticks',[0.4,.62,.88],'TickLabels',[]);%'TickLabels',{'Low','Medium','High'}');
yh = ylabel(ax5.Colorbar,'Low       Medium       High'); set(yh,'Rotation',90);
colorbar off;
 
subplot(2,4,6);
cmap = jet(length(edges));
for i = 1:length(edges)-1
   h = bar(i,texture_vec(i),'facecolor',cmap(i+1,:),'edgecolor','k');set(h,'BarWidth',1);hold on;
end
set(gca,'Ylim',[0 1],'box','off','Xtick',1:length(edges)-1,'Fontname','Timesnewroman','Fontsize',12,'XtickLabel',{'L','M','H'},'Linewidth',2);%'XtickLabel',{'Low','Medium','High'});
title('Textural');
ylabel('Relative Distribution of Expression Levels');
xlabel('Feature Expression Levels');
 
subplot(2,4,7);
h = bar(spatial_vec,'facecolor','m','edgecolor','k');set(h,'BarWidth',1);
set(gca,'Ylim',[0 1],'box','off','Fontname','Timesnewroman','Fontsize',12,'Linewidth',2,'XTickLabel',{'L<->M','L<->H','M<->H'},'Fontsize',12);
title('Spatial');
ylabel('Relative Frequency','Fontsize',14);
xlabel('Feature Expression Edge Pairs','Fontsize',14);


end

