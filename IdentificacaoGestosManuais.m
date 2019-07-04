%Cria o caminho até a pasta com as imagens do banco de dados;
PastaSaida = fullfile('handgesture');
PastaRaiz = fullfile(PastaSaida,'HandGestureCategories');

%Define as classes;
Classes = {'A', 'B', 'C', 'I', 'K', 'L', 'O' 'R', ...
    'S', 'U', 'W', 'Y'};

%Armazena a localização das imagens;
ImageDataStore = imageDatastore(fullfile(PastaRaiz,Classes),'LabelSource', ... 
    'foldernames');

%Obtém a quantidade mínima de imagens em todas as pastas de classes;
Tabela = countEachLabel(ImageDataStore);
MinTabela = min(Tabela{:,2});

%Atualiza ImageDataStore para que cada classe tenha apenas o mínimo de
%imagens encontradas anteriormente;
ImageDataStore = splitEachLabel(ImageDataStore, MinTabela, 'randomize');

%Carrega a CNN pré-treinada (resnet50);
Rede = resnet50;

%Prepara os conjuntos de treinamento e de teste, onde 40% das imagens serão
%para treinamento e 60% para teste (as imagens são escolhidas aleatoriamente);
[ConjuntoTreinamento, ConjuntoTeste] = splitEachLabel(ImageDataStore, 0.4, ...
    'randomize');

%Obtém o tamanho necessário para as imagens de entrada;
TamImagem = Rede.Layers(1).InputSize;

%Redimensiona as imagens dos conjuntos para o tamanho encontrado
%anteriormente, além disso converte imagens em níveis de cinza para imagens
%RGB;
ConjuntoTreinamentoAumentado = augmentedImageDatastore(TamImagem, ... 
    ConjuntoTreinamento, 'ColorPreprocessing', 'gray2rgb');

ConjuntoTesteAumentado = augmentedImageDatastore(TamImagem, ConjuntoTeste, ... 
    'ColorPreprocessing', 'gray2rgb');

%Extrai as características do conjunto de treinamento;
Camada_Caracteristicas = 'fc1000';
Caracteristicas_Treinamento = activations(Rede, ConjuntoTreinamentoAumentado, ...
    Camada_Caracteristicas, 'MiniBatchSize', 32, 'OutputAs', 'columns');

%Classifica as imagens do conjunto de treinamento;
ClasseTreinamento = ConjuntoTreinamento.Labels;
Classificador = fitcecoc(Caracteristicas_Treinamento, ClasseTreinamento, ...
    'Learner', 'Linear', 'Coding', 'onevsall', 'ObservationsIn', 'columns');

%Extrai as características do conjunto de teste; 
Caracteristicas_Teste = activations(Rede, ConjuntoTesteAumentado, ... 
    Camada_Caracteristicas, 'MiniBatchSize', 32, 'OutputAs', 'columns');

%Mostra a tabela de confusão.
Predicao = predict(Classificador, Caracteristicas_Teste, 'ObservationsIn', 'columns');
plotconfusion(ConjuntoTeste.Labels, Predicao)
