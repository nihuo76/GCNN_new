%%
myFolder = './hamiltonMER';
filePattern = fullfile(myFolder, '*.mat');
matFiles = dir(filePattern);
percentVec = zeros(length(matFiles),1);

%%
for k = 1:length(matFiles)
  baseFileName = matFiles(k).name;
  fullFileName = fullfile(myFolder, baseFileName);
  %fprintf(1, 'Now reading %s\n', fullFileName);
  test_data = load(fullFileName);
  test_position = test_data.Hr.cell_position;
  test_matrix = test_data.Hr.Ham;
  y = find(abs(test_position(:,1)) >= 2 | ...
      abs(test_position(:,2)) >= 2 | ...
      abs(test_position(:,3)) >= 2);
  zeroHam = sum(sum(sum( abs(real( test_matrix(:,:,y) )) <0.001 )));
  percentVec(k) = zeroHam / (numel(test_matrix(:,:,y)));
  %display(test_data.Hr.norb)
  %edges = [-1:0.001:1];
  %histogram(real(test_matrix(:,:,y)), edges)
  %numel(test_matrix)
end
histogram(percentVec)
%%
decision_bound = 0.2;
countBG = zeros(length(matFiles),1);
countflag=0;
for k = 1:length(matFiles)
  baseFileName = matFiles(k).name;
  fullFileName = fullfile(myFolder, baseFileName);
  %fprintf(1, 'Now reading %s\n', fullFileName);
  test_data = load(fullFileName);
  bgap = test_data.Hr.band_gap;
  if bgap <= decision_bound
      countflag = countflag+1;
  end
  countBG(k) = bgap;
  %test_matrix = test_data.Hr.Ham;
%   y = find(abs(test_position(:,1)) >= 2 | ...
%       abs(test_position(:,2)) >= 2 | ...
%       abs(test_position(:,3)) >= 2);
%   zeroHam = sum(sum(sum( abs(real( test_matrix(:,:,y) )) <0.001 )));
%   percentVec(k) = zeroHam / (numel(test_matrix(:,:,y)));
%   display(test_data.Hr.norb)
%   edges = [-1:0.001:1];
%   histogram(real(test_matrix(:,:,y)), edges)
%   numel(test_matrix)
end
% edges = [0:0.01:10];
% histogram(countBG, edges);
display(decision_bound);
countflag/k