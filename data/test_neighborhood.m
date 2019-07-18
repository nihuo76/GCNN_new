%%
myFolder = './hamiltonMER';
filePattern = fullfile(myFolder, '*.mat');
matFiles = dir(filePattern);
percentVec = zeros(length(matFiles),1);
countOrb = zeros(length(matFiles),1);
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
for k = 1:length(matFiles)
  baseFileName = matFiles(k).name;
  fullFileName = fullfile(myFolder, baseFileName);
  %fprintf(1, 'Now reading %s\n', fullFileName);
  test_data = load(fullFileName);
  numorb = test_data.Hr.norb;
  countOrb(k) = numorb;
  %test_matrix = test_data.Hr.Ham;
%   y = find(abs(test_position(:,1)) >= 2 | ...
%       abs(test_position(:,2)) >= 2 | ...
%       abs(test_position(:,3)) >= 2);
%   zeroHam = sum(sum(sum( abs(real( test_matrix(:,:,y) )) <0.001 )));
%   percentVec(k) = zeroHam / (numel(test_matrix(:,:,y)));
  %display(test_data.Hr.norb)
  %edges = [-1:0.001:1];
  %histogram(real(test_matrix(:,:,y)), edges)
  %numel(test_matrix)
end
histogram(countOrb)