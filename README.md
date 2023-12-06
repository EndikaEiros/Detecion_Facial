# Detección y reconocimiento facial


## Modo de uso:

### Data.py

- Introducir nueva cara

    - Opción 1: Obtener imágenes del rostro desde un vídeo:
    ```
    generate_data_as_images(path_to_input_video, person_name, path_to_output_images)
    ```
    - Opción 2: Obtener landmarks del rostro desde un vídeo:
    ```
    generate_data_as_images(path_to_input_video, person_name, path_to_output_landmarks)
    ```
  
### Model.py

- Cargar modelo previamente entrenado
    ```
    save_model(trained_model, name_of_saved_model)
    ```
- Guardar modelo entrenado
    ```
    trained_model = load_model(path_to_trained_model)
    ```
- Cargar imágenes para entrenar o evaluar
    ```
    X_train, y_train = load_image_data(path_to_directory_of_images)
    ```
- Evaluar modelo
    ```
    accuracy = test_model(trained_model, path_to_test_images)
    ```
- Predecir una instancia
    ```
    prediction = predict_one(trained_model, path_to_an_image)
    ```
- Entrenar modelo
    ```
    trained_model = train_model(path_to_train_images, new_model)
    ```
  
### Recognition.py

- Obtener todas las caras de un frame
    ```
    faces = get_faces(frame, optional_angles)
    ```
- Dibujar sobre los rostros de un frame
  ```
  drawed_frame = draw_square(frame, x_coord, y_coord, height, weight, person_name)
  ```
  