function spatial_vec = buildSpatialVec(clusteredImg,binedges,graphstruct)
% Function for computing the spatial component of RADISTAT

%assumes that clusteredImg ranges from 0 - 1 with NaNs outside ROI

%% TEST 2D IMAGE
%disp('TESTING');
% y = [1/3 1/3 1/3; 1/3 1/3 1/3; 1/3 1/3 1/3];
% 
% clusteredImg = y;
% binedges = [0 1/3 2/3 1];
% graphstruct = 'adjacency';

%% TEST 3D IMAGE
%disp('TESTING');
% y(:,:,1) = [1/3 1/3 1/3; 1/3 1/3 1/3; 1/3 1/3 1/3];
% y(:,:,2) = [1/3 1/3 1/3; 1/3 3/3 1/3; 1/3 1/3 1/3];
% y(:,:,3) = [1/3 1/3 1/3; 1/3 1/3 1/3; 2/3 1/3 1/3];
% 
% clusteredImg = y;
% binedges = [0 1/3 2/3 1];
% graphstruct = 'adjacency';

%% extract permiter pixels of all expression level groups
fprintf('\tbuilding spatial vec...');

% prepare image
u = unique(clusteredImg);
u(isnan(u)) = [];   
u = sort(u,'ascend');
centroids = [];
labels = [];
bins = binedges(2:end);
for i = 1:length(u)
    currExpr = u(i);
    idx = find(abs(bins-currExpr)<1e-2); %doesn't always work comparing double to double in MATLAB
    if isempty(idx)
        error('Unable to match current expression level to any bin values.');
    end
    BW = (clusteredImg==currExpr);
    CC = bwconncomp(BW);
    ExprStruct(idx).Connectivity = CC.Connectivity;
    ExprStruct(idx).ImageSize = CC.ImageSize;
    ExprStruct(idx).NumObjects = CC.NumObjects;
    ExprStruct(idx).PixelIdxList = CC.PixelIdxList;
    for j = 1:ExprStruct(idx).NumObjects
        clear temp
        temp = false(ExprStruct(idx).ImageSize);
        temp(ExprStruct(idx).PixelIdxList{j})=true;
        perim = find(bwperim(temp)>0);
        if isempty(perim)
            ExprStruct(idx).PerimIdxList{j}=ExprStruct(idx).PixelIdxList{j};
        else
            ExprStruct(idx).PerimIdxList{j}=perim;
        end
    end
    stats = regionprops(BW,'centroid');
    centroids = [centroids; cat(1,stats.Centroid)];
    labels = [labels;repmat(u(i),length(stats),1)];
end
clear CC perim stats temp i j

spatial_vec = [];

%% 2D
if numel(size(clusteredImg))==2 %FOR 2D IMAGES
    x = centroids(:,1);
    y = centroids(:,2);
        
    if strcmp(graphstruct,'adjacency')
      % ADJACENCY  
      nbins = length(binedges)-1;
      adj = zeros(nbins);
      for i = 1:length(ExprStruct)
          skipIdx = i;
          for j = 1:ExprStruct(i).NumObjects
              perim = ExprStruct(i).PerimIdxList{j};
              neighbors = [];
              levels = [];
              groups = [];
              for k = 1:length(perim)
                  %check all sides of perim
                  [r, c] = ind2sub(size(clusteredImg),perim(k));
                  [l, g] = identifyExprGroup(ExprStruct,skipIdx,r+1,c); % below
                  levels = [levels l]; groups = [groups g];
                  [l, g] = identifyExprGroup(ExprStruct,skipIdx,r+1,c+1); % bottom right
                  levels = [levels l]; groups = [groups g];
                  [l, g] = identifyExprGroup(ExprStruct,skipIdx,r,c+1); % right
                  levels = [levels l]; groups = [groups g];
                  [l, g] = identifyExprGroup(ExprStruct,skipIdx,r-1,c+1); % upper right
                  levels = [levels l]; groups = [groups g];
                  [l, g] = identifyExprGroup(ExprStruct,skipIdx,r-1,c); % above
                  levels = [levels l]; groups = [groups g];
                  [l, g] = identifyExprGroup(ExprStruct,skipIdx,r-1,c-1); %upper left
                  levels = [levels l]; groups = [groups g];
                  [l, g] = identifyExprGroup(ExprStruct,skipIdx,r,c-1); %left
                  levels = [levels l]; groups = [groups g];
                  [l, g] = identifyExprGroup(ExprStruct,skipIdx,r+1,c-1); %bottom left
                  levels = [levels l]; groups = [groups g];
              end
              neighbors = unique([levels;groups]','rows');
              for k = 1:size(neighbors,1)
                  adj(skipIdx,neighbors(k,1)) = adj(skipIdx,neighbors(k,1))+1;
              end
          end
      end
      if ~issymmetric(adj)
          error('Adjacency matrix not symmetric. buildSpatialVec.m needs debugging');
      end
      uniquevec = reshape(adj,[1 nbins^2]);
      spatial_vec(1) = uniquevec(4)+uniquevec(2);
      spatial_vec(2) = uniquevec(7)+uniquevec(3);
      spatial_vec(3) = uniquevec(6)+uniquevec(8);
      
      %normalize
      nedges = sum(spatial_vec(:));
      if nedges ~= 0
          spatial_vec = spatial_vec/(nedges);
      end

  elseif strcmp(graphstruct,'mst')
        %% MIN SPAN TREE -- UNDIRECTED
        A = ones(size(centroids,1));%adjaceny matrix
%         assign weights based on distance
        for i = 1:size(centroids,1)
            for j = 1:size(centroids,1);
              A(i,j) = sqrt((centroids(i,1)-centroids(j,1))^2+(centroids(i,2)-centroids(j,2))^2);
            end
        end
        G = graph(A);
        minT = minspantree(G);
        minA = adjacency(minT);
        [s_minG,t_minG,w] = find(minA); %get ith node, jth node of an edge and the edge weight;
        %optional -- view MST
        figure('Color','white');
        subplot(1,2,1);
        imagesc(clusteredImg);colormap();cmap = jet(length(u)+1);cmap(1,:) = [0 0 0];colormap(cmap);caxis([0 max(bins)]);colorbar;
        hold on;
        for k = 1:length(s_minG)
            x1=centroids(s_minG(k),1);y1 = centroids(s_minG(k),2);
            x2=centroids(t_minG(k),1);y2 = centroids(t_minG(k),2);
            h=plot([x1 x2],[y1 y2],'m-o','Linewidth',2,'Markersize',12);
        end
        title('MinSpanTree');axis off;
        hold off;
%         [uniquevec,lumpedvec] =
%         spatialVec_2D(s_minG,t_minG,labels,binedges); %dont know what to do yet
        
        %normalize
%         nedges = length(s_minG);
%         
%         if nedges > 0
%             uniquevec = uniquevec/(nedges);
%             lumpedvec = lumpedvec/(nedges);
%         end
%         subplot(1,2,2);
%         h = bar(lumpedvec,'facecolor','m','edgecolor','k');set(h,'BarWidth',1);
%         set(gca,'Ylim',[0 1],'XtickLabel',[],'FontSize',12,'Fontname','Timesnewroman');
        %'XTickLabel',{'L<->L','L<->M','L<->H','M<->M','M<->H','H<->H'},'Fontsize',12);
%         ylabel('Relative Frequency','Fontsize',14);title('Undirected Graph: Lumped Directions','Fontsize',16,'Fontweight','bold');xlabel('Feature Expression Edge Pairs','Fontsize',14);
       spatial_vec = [];
   elseif strcmp(graphstruct,'delauney')
       %Delauney Triangulation
       if length(x)<3
            fprintf('Unable to perform Delauney triangulation. At least 3 centroids needed\n');
            lumpedvec = [];
       else
            tri = delaunay(x,y);

    %         convert to [s,t] form, where (s,t) are nodes connected by an edge, and s is the starting node and t is the ending node
            %% UNDIRECTED GRAPH
            s = []; t = [];
            for i = 1:size(tri,1)
                A = tri(i,1); B = tri(i,2); C = tri(i,3);
                s = [s A]; t = [t B]; %(A->B)
                s = [s B]; t = [t A]; %(B->A)
                s = [s A]; t = [t C]; %(A->C)
                s = [s C]; t = [t A]; %(C->A)
                s = [s B]; t = [t C]; %(B->C)
                s = [s C]; t = [t B]; %(C->B)
            end
            s_undG = s;
            t_undG = t;

%             [uniquevec, lumpedvec] = spatialVec_2D(s_undG,t_undG,labels,binedges);

            %normalize
%             nedges = length(s_undG);
% 
%             uniquevec = uniquevec/(nedges);
%             lumpedvec = lumpedvec/(nedges);

    %         %plot results -- YES!
            figure('Color','white');
            subplot(1,2,1);
            imagesc(clusteredImg);colormap();cmap = jet(length(u)+1);cmap(1,:) = [0 0 0];colormap(cmap);caxis([0 max(clusteredImg(:))]);colorbar;
            hold on;
            h = triplot(tri,x,y,'m');
            set(h,'Linewidth',2);set(gca,'Ydir','reverse');axis off;
            title('Delauney');
            hold off;
%             subplot(1,3,2);
%             h = bar(uniquevec,'facecolor','m','edgecolor','k');
%             ylabel('Frequency');title('Undirected Graph: Unique directions');
%             subplot(1,2,2);
%             h = bar(lumpedvec,'facecolor','m','edgecolor','k');
%             set(gca,'Ylim',[0 1],'XTickLabel',{'L<->L','L<->M','L<->H','M<->M','M<->H','H<->H'},'Fontsize',12);
%             ylabel('Relative Frequency','Fontsize',14);title('Undirected Graph: Lumped Directions','Fontsize',16,'Fontweight','bold');xlabel('Feature Expression Edge Pairs','Fontsize',14);
%             close gcf;
            spatial_vec = [];
       end
   else
       error('invalid graphstruct parameter');
    end
      
%% 3D
elseif numel(size(clusteredImg))==3 %FOR 3D IMAGES
    x = centroids(:,1);
    y = centroids(:,2);
    
   if strcmp(graphstruct,'adjacency')
      % ADJACENCY  
      nbins = length(binedges)-1;
      adj = zeros(nbins);
      for i = 1:length(ExprStruct)
          skipIdx = i;
          for j = 1:ExprStruct(i).NumObjects
              perim = ExprStruct(i).PerimIdxList{j};
              neighbors = [];
              levels = [];
              groups = [];
              for k = 1:length(perim)
                  %check all sides of perim
                  [r, c, s] = ind2sub(size(clusteredImg),perim(k));
                  %----------same slice------------%
                  [l, g] = identifyExprGroup(ExprStruct,skipIdx,r+1,c,s); % below
                  levels = [levels l]; groups = [groups g];
                  [l, g] = identifyExprGroup(ExprStruct,skipIdx,r+1,c+1,s); % bottom right
                  levels = [levels l]; groups = [groups g];
                  [l, g] = identifyExprGroup(ExprStruct,skipIdx,r,c+1,s); % right
                  levels = [levels l]; groups = [groups g];
                  [l, g] = identifyExprGroup(ExprStruct,skipIdx,r-1,c+1,s); % upper right
                  levels = [levels l]; groups = [groups g];
                  [l, g] = identifyExprGroup(ExprStruct,skipIdx,r-1,c,s); % above
                  levels = [levels l]; groups = [groups g];
                  [l, g] = identifyExprGroup(ExprStruct,skipIdx,r-1,c-1,s); %upper left
                  levels = [levels l]; groups = [groups g];
                  [l, g] = identifyExprGroup(ExprStruct,skipIdx,r,c-1,s); %left
                  levels = [levels l]; groups = [groups g];
                  [l, g] = identifyExprGroup(ExprStruct,skipIdx,r+1,c-1,s); %bottom left
                  levels = [levels l]; groups = [groups g];
                  %----------back one slice------------%
                  [l, g] = identifyExprGroup(ExprStruct,skipIdx,r+1,c,s-1); % below
                  levels = [levels l]; groups = [groups g];
                  [l, g] = identifyExprGroup(ExprStruct,skipIdx,r+1,c+1,s-1); % bottom right
                  levels = [levels l]; groups = [groups g];
                  [l, g] = identifyExprGroup(ExprStruct,skipIdx,r,c+1,s-1); % right
                  levels = [levels l]; groups = [groups g];
                  [l, g] = identifyExprGroup(ExprStruct,skipIdx,r-1,c+1,s-1); % upper right
                  levels = [levels l]; groups = [groups g];
                  [l, g] = identifyExprGroup(ExprStruct,skipIdx,r-1,c,s-1); % above
                  levels = [levels l]; groups = [groups g];
                  [l, g] = identifyExprGroup(ExprStruct,skipIdx,r-1,c-1,s-1); %upper left
                  levels = [levels l]; groups = [groups g];
                  [l, g] = identifyExprGroup(ExprStruct,skipIdx,r,c-1,s-1); %left
                  levels = [levels l]; groups = [groups g];
                  [l, g] = identifyExprGroup(ExprStruct,skipIdx,r+1,c-1,s-1); %bottom left
                  levels = [levels l]; groups = [groups g];
                 %----------forward one slice------------%
                  [l, g] = identifyExprGroup(ExprStruct,skipIdx,r+1,c,s+1); % below
                  levels = [levels l]; groups = [groups g];
                  [l, g] = identifyExprGroup(ExprStruct,skipIdx,r+1,c+1,s+1); % bottom right
                  levels = [levels l]; groups = [groups g];
                  [l, g] = identifyExprGroup(ExprStruct,skipIdx,r,c+1,s+1); % right
                  levels = [levels l]; groups = [groups g];
                  [l, g] = identifyExprGroup(ExprStruct,skipIdx,r-1,c+1,s+1); % upper right
                  levels = [levels l]; groups = [groups g];
                  [l, g] = identifyExprGroup(ExprStruct,skipIdx,r-1,c,s+1); % above
                  levels = [levels l]; groups = [groups g];
                  [l, g] = identifyExprGroup(ExprStruct,skipIdx,r-1,c-1,s+1); %upper left
                  levels = [levels l]; groups = [groups g];
                  [l, g] = identifyExprGroup(ExprStruct,skipIdx,r,c-1,s+1); %left
                  levels = [levels l]; groups = [groups g];
                  [l, g] = identifyExprGroup(ExprStruct,skipIdx,r+1,c-1,s+1); %bottom left
                  levels = [levels l]; groups = [groups g];
              end
              neighbors = unique([levels;groups]','rows');
              for k = 1:size(neighbors,1)
                  adj(skipIdx,neighbors(k,1)) = adj(skipIdx,neighbors(k,1))+1;
              end
              
          end
      end
      if ~issymmetric(adj)
          error('Adjacency matrix not symmetric. buildSpatialVec.m needs debugging');
      end
      uniquevec = reshape(adj,[1 nbins^2]);
      spatial_vec(1) = uniquevec(4)+uniquevec(2);
      spatial_vec(2) = uniquevec(7)+uniquevec(3);
      spatial_vec(3) = uniquevec(6)+uniquevec(8);
      
      %normalize
      nedges = sum(spatial_vec(:));
      if nedges ~= 0
          spatial_vec = spatial_vec/(nedges);
      end
   else
       error('invalid graphstruct parameter');
   end
       
else
    error('invalid image size. must be 2D or 3D');
end

% %plot -- testing
% figure('Color','white')
% subplot(1,2,2);
% h = bar(spatial_vec,'facecolor','m','edgecolor','k');set(h,'BarWidth',1);
% set(gca,'Ylim',[0 1],'box','off','Fontname','Timesnewroman','Fontsize',12,'XtickLabel',[],'XTickLabel',{'L<->M','L<->H','M<->H'},'Fontsize',12);
% subplot(1,2,1);
% cmap = jet(4);cmap(1,:) = [0 0 0];
% colormap(cmap);caxis([0 1]);
% vv(clusteredImg,[],cmap,[0 1]);

fprintf('done.\n');


end

%% Subfunctions
% function [uniquevec, lumpedvec] = spatialVec(s,t,labels,binedges)
% nbins = length(binedges)-1;
% uniquevec = zeros(nbins);  
% % POSSIBLE OUTCOMES & how they are stored
% %     C   B  A
% %  C->                        (1,1) (1,2) (1,3)
% %  B->                <===>   (2,1) (2,2) (2,3)
% %  A->                        (3,1) (3,2) (3,3)
% for i = 1:length(s)
%     j = find(binedges==labels(s(i)))-1;%left most binedge can be ignored
%     k = find(binedges==labels(t(i)))-1;
%     try
%     uniquevec(j,k) = uniquevec(j,k)+1;
%     catch
%         x=1;
%     end
% end
% 
% uniquevec = reshape(uniquevec,[1 nbins^2]);
% 
% %OR you double count (i.e. A<-->B gets the same assignment)
% lumpedvec(1) = uniquevec(1);
% lumpedvec(2) = uniquevec(2) + uniquevec(4);
% lumpedvec(3) = uniquevec(3) + uniquevec(7);
% lumpedvec(4) = uniquevec(5);
% lumpedvec(5) = uniquevec(6)+uniquevec(8);
% lumpedvec(6) = uniquevec(9);
% end

function [level, group] = identifyExprGroup(ExprStruct,skipIdx,r,c,s)
%s: only needed if 3D image

level = []; group = [];

for i = 1:length(ExprStruct) 
  
    %check that we aren't looking at same expression group
    if i == skipIdx
        continue;
    end
    
    %check that this expression level exists
    if isempty(ExprStruct(i).Connectivity)
        continue;
    end
 
    
    %check that we haven't crossed the image border
    if r>ExprStruct(i).ImageSize(1) || c>ExprStruct(i).ImageSize(2) || r<1 || c<1 
            return;
    end
    if numel(ExprStruct(i).ImageSize)==3 %% additional check for 3D image
        if s>ExprStruct(i).ImageSize(3) || s<1
            return;
        end
    end

    %2D or 3D
    if numel(ExprStruct(i).ImageSize) == 2
        pix = sub2ind(ExprStruct(i).ImageSize,r,c);
    elseif numel(ExprStruct(i).ImageSize) == 3
        pix = sub2ind(ExprStruct(i).ImageSize,r,c,s);
    end
    
    for j = 1:ExprStruct(i).NumObjects
        if ~isempty(find(ExprStruct(i).PerimIdxList{j}==pix,1)) %is this a perimeter pixel of a different expression level group?? let's record it!
            level = i; %i = expression level
            group = j; %j = which clustered object
            return;
        end
    end
end
end
