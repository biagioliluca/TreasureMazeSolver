import manage_dataset
import model

save_dataset_path = ""

if __name__ == '__main__':
	try:
		training_dtset, training_labels = import_dataset(save_dataset_path + 'training_dataset.csv')
		test_dtset, test_labels = import_dataset(save_dataset_path + 'testing_dataset.csv')
	except:
		create_dataset(save_dataset_path)
		training_dtset, training_labels = import_dataset(save_dataset_path + 'training_dataset.csv')
		test_dtset, test_labels = import_dataset(save_dataset_path + 'testing_dataset.csv')

	training_dtset, test_dtset = dataset_normalization(training_dtset, test_dtset)

	epochs = 15
	batch_values = [64]

	model = create_model(num_classes)
    model.compile(loss ="sparse_categorical_crossentropy",optimizer = "adam", metrics=["accuracy"])
    history = model.fit(training_images, training_labels, batch_size= batch_size, epochs= epochs)
    score = model.evaluate(test_images, test_labels, verbose=0)
    my_score = grid_evaluation(model, grid_path, grids[grid_name])
    output = output + "Model with batch size = {} and epochs = {}\n\tTest loss: {}\n\tTest accuracy: {}\n".format(batch_size, epochs, score[0], score[1])
    output = output + "\tDigits correctly classified: {}\n".format(my_score)
    model.reset_metrics()

    print(output)
