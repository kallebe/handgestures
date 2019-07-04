%Abre a imagem;
Imagem = imread('img_teste\o.jpg');

%Extrai as caracter�sticas da imagem;
DataStore = augmentedImageDatastore(TamImagem, Imagem, ... 
    'ColorPreprocessing', 'gray2rgb');

Caracteristicas_Imagem = activations(Rede, DataStore, Camada_Caracteristicas, ... 
    'MiniBatchSize', 32, 'OutputAs', 'columns');

%Define � qual classe a imagem pertence por meio da CNN criada;
Classe = predict(Classificador, Caracteristicas_Imagem, 'ObservationsIn', ...
    'columns');

%Mostra imagem
figure, imshow(Imagem);

%Abre uma janela com a mensagem que indica o s�mbolo presente na imagem.
mensagem =  sprintf('A imagem corresponde ao s�mbolo %s.', Classe);
CaixaDialogo = msgbox(mensagem, 'Classifica��o');
set(CaixaDialogo, 'position', [450 350 270 60]);
aux = get(CaixaDialogo, 'CurrentAxes');
aux = get(aux, 'Children');
set(aux, 'FontSize', 15);
