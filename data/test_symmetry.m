%%
myFolder = 'C:\Users\Hexin\Documents\\GCNN_new\data\hamiltonMER';
filePattern = fullfile(myFolder, '*.mat');
matFiles = dir(filePattern);
%%
count_flag = 0;
for k=1:length(matFiles)
    baseFileName = matFiles(k).name;
    fullFileName = fullfile(myFolder, baseFileName);
    %testFile = matFiles(3).name;
    %testFullname = fullfile(myFolder, testFile);
    test_data = load(fullFileName);
    test_conduct = test_data.Hr.band_gap;
    if (test_conduct >= 0.2)
        count_flag = count_flag+1;
    end
end
display(count_flag/233)
%%
all_343 = 1;
for k = 1:length(matFiles)
  baseFileName = matFiles(k).name;
  fullFileName = fullfile(myFolder, baseFileName);
  %fprintf(1, 'Now reading %s\n', fullFileName);
  testFile = matFiles(3).name;
  testFullname = fullfile(myFolder, testFile);
  test_data = load(testFullname);
  test_position = test_data.Hr.cell_position;
  if (length(test_position) ~= 343)
      all_343 = 0;
  end
end
all_343
%%
% test symmetry
testFile = matFiles(3).name;
testFullname = fullfile(myFolder, testFile);
test_data = load(testFullname);
test_Hamiton = test_data.Hr.Ham;
test_position = test_data.Hr.cell_position;
copy_position = test_position;
%%

for j = 1:length(test_position)
    x = test_position(j , :);
    y = find(-x(:,1) == test_position(:, 1) & -x(:,2) == test_position(:, 2) & -x(:,3) == test_position(:,3));
    j
    prod(prod(real(test_Hamiton(:,:,j) - test_Hamiton(:,:,y)) == 0))
end


