dataset = imageDatastore('/Users/roshankandel/Downloads/untitled folder/','IncludeSubfolders',true,'LabelSource','foldernames');
tbl = countEachLabel(dataset);
disp(tbl)
figure
montage(dataset.Files(1:16:end))

[trainingSet, validationSet] = splitEachLabel(dataset, 0.6, 'randomize');
bag = bagOfFeatures(trainingSet);

img = readimage(dataset, 1);
featureVector = encode(bag, img);

% Plot the histogram of visual word occurrences
figure
bar(featureVector)
title('Visual word occurrences')
xlabel('Visual word index')
ylabel('Frequency of occurrence')

categoryClassifier = trainImageCategoryClassifier(trainingSet, bag);

% Evaluate the classifier on the training set
confMatrix_train = evaluate(categoryClassifier, trainingSet);


categoryClassifier_val = trainImageCategoryClassifier(validationSet, bag);

% Evaluate the classifier on the validation set
confMatrix_val = evaluate(categoryClassifier_val, validationSet);




