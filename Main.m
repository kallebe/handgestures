%Abre a imagem;
Imagem = imread('img_teste\o.jpg');

%Extrai as características da imagem;
DataStore = augmentedImageDatastore(TamImagem, Imagem, ... 
    'ColorPreprocessing', 'gray2rgb');

Caracteristicas_Imagem = activations(Rede, DataStore, Camada_Caracteristicas, ... 
    'MiniBatchSize', 32, 'OutputAs', 'columns');

%Define à qual classe a imagem pertence por meio da CNN criada;
Classe = predict(Classificador, Caracteristicas_Imagem, 'ObservationsIn', ...
    'columns');

%Mostra imagem
figure, imshow(Imagem);

%Abre uma janela com a mensagem que indica o símbolo presente na imagem.
mensagem =  sprintf('A imagem corresponde ao símbolo %s.', Classe);
CaixaDialogo = msgbox(mensagem, 'Classificação');
set(CaixaDialogo, 'position', [450 350 270 60]);
aux = get(CaixaDialogo, 'CurrentAxes');
aux = get(aux, 'Children');
set(aux, 'FontSize', 15);
