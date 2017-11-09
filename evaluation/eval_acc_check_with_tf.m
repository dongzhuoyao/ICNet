function eval_acc()
addpath('evaluationCode');
addpath('visualizationCode');
load(data_class);

root_dir = '/home/hut/lab/tensorpack/examples/UNet/result/';


%% Evaluation
% initialize statistics
cnt=0;
area_intersection = double.empty;
area_union = double.empty;
pixel_accuracy = double.empty;
pixel_correct = double.empty;
pixel_labeled = double.empty;

% main loop
for i = 0: 705
    filePred = root_dir + i +'-predict.jpg';
    fileAnno = root_dir + i + '-label.jpg';
    % read in prediction and label
    imPred = imread(filePred);
    imAnno = imread(fileAnno);
    

    % compute IoU
    cnt = cnt + 1;
    [area_intersection(:,cnt), area_union(:,cnt)] = intersectionAndUnion(imPred, imAnno, numClass);

    % compute pixel-wise accuracy
    [pixel_accuracy(i), pixel_correct(i), pixel_labeled(i)] = pixelAccuracy(imPred, imAnno);
    fprintf('Evaluating %d/%d... Pixel-wise accuracy: %2.2f%%\n', cnt, numel(filesPred), pixel_accuracy(i)*100.);
end

%% Summary
IoU = sum(area_intersection,2)./sum(eps+area_union,2);
mean_IoU = mean(IoU);
accuracy = sum(pixel_correct)/sum(pixel_labeled);

fprintf('==== Summary IoU ====\n');
for i = 1:numClass
    fprintf('class %3d iou : %.4f\n', i, IoU(i));
end
fprintf('Mean IoU over %d classes: %2.2f%%\n', numClass, mean_IoU*100.);
fprintf('Pixel-wise Accuracy: %2.2f%%\n', accuracy*100.);
