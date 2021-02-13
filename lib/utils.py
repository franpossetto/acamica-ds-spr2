class utils:
    @staticmethod
    def open_csv(path):
        return pd.read_csv(path)

    @staticmethod
    def graph_by_property_value(property_name, property_value, order_array, ax1,ax2):
        a = sns.countplot(data = dataset[dataset.l2 == property_value],y=property_name, palette="Set3",ax=ax[ax1,ax2], order=order_array)
        a.set_title(property_value)
    
    @staticmethod
    def split(features,obj,test_size):
        X_train, X_test, Y_train, Y_test = train_test_split(features, obj, 
                                                            test_size=test_size, 
                                                            random_state=42)
        return X_train, X_test, Y_train, Y_test
    
    @staticmethod
    def rmse_graph(model, X_train, X_test, Y_train, Y_test):
        Y_train_pred = model.predict(X_train)
        Y_test_pred = model.predict(X_test)
        
        rmse_train = np.sqrt(mean_squared_error(Y_train, Y_train_pred))
        rmse_test = np.sqrt(mean_squared_error(Y_test, Y_test_pred))
        
        print(f'Err Train: {rmse_train}')
        print(f'Err Test: {rmse_test}')

        plt.figure(figsize = (10,4))
        
        sns.distplot(Y_train - Y_train_pred, bins = 20, label = 'train')
        sns.distplot(Y_test - Y_test_pred, bins = 20, label = 'test')
        
        plt.xlabel('Errores')
        plt.legend()

    
    @staticmethod
    def decision_tree(title, depths, X_train, X_test, Y_train, Y_test):
        #Define error list
        err_train = []
        err_test = []
        for depth in depths:
            
            #Define DesitionTreeRegresson and fit it
            a = DecisionTreeRegressor(max_depth = depth, random_state = 42)
            a.fit(X_train, Y_train)
            
            #try
            Y_train_pred = a.predict(X_train)
            train_err = np.sqrt(mean_squared_error(Y_train, Y_train_pred))
            
            Y_test_pred = a.predict(X_test)
            test_err = np.sqrt(mean_squared_error(Y_test, Y_test_pred))

            #save the error
            err_train.append(train_err)
            err_test.append(test_err)
         
        plt.plot(depths, err_train,'o-',label='train' )
        plt.plot(depths, err_test, 'o-',label='test')
        plt.legend()
        plt.title(title)
        plt.xlabel('depths')
        plt.ylabel('error')
    

    @staticmethod
    def knn(titulo, k_vecinos, X_train, X_test, y_train, y_test):
        # Definimos las listas vacias para los valores de error deseados
        lista_error_train = []
        lista_error_test = []

        for k in k_vecinos:
            # Definir el modelo con el valor de vecinos deseado
            clf = KNeighborsRegressor(n_neighbors= k)

            # Entrenar el modelo
            clf.fit(X_train, y_train)

            # Predecir y evaluar sobre el set de entrenamiento
            y_train_pred = clf.predict(X_train)
            train_err = np.sqrt(mean_squared_error(y_train, y_train_pred))

            # Predecir y evaluar sobre el set de evaluación
            y_test_pred = clf.predict(X_test)
            test_err = np.sqrt(mean_squared_error(y_test, y_test_pred))

            # Agregar la información a las listas
            lista_error_train.append(train_err)
            lista_error_test.append(test_err)

        plt.plot(k_vecinos, lista_error_train,'o-',label='train' )
        plt.plot(k_vecinos, lista_error_test, 'o-',label='test')
        plt.legend()
        plt.title(titulo)
        plt.xlabel('k_vecinos')
        plt.ylabel('error')
        plt.ylabel('error')