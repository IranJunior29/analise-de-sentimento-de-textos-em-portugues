# Imports
import csv
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split

# Classe de tokenizador dos dados
class SentimentAnalysisTokenizer(Dataset):

    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):

        text = self.texts[idx]
        label = self.labels[idx]
        inputs = self.tokenizer.encode_plus(text,
                                            add_special_tokens=True,
                                            max_length=self.max_length,
                                            padding='max_length',
                                            truncation=True,
                                            return_tensors='pt')

        return {
            'input_ids': inputs['input_ids'].squeeze(0),
            'attention_mask': inputs['attention_mask'].squeeze(0),
            'label': torch.tensor(label)
        }

if __name__ == '__main__':

    ''' Verificando o Ambiente de Desenvolvimento '''

    # Verifica se uma GPU está disponível e define o dispositivo apropriado
    processing_device = "cuda" if torch.cuda.is_available() else "cpu"

    # Define o device (GPU ou CPU)
    device = torch.device(processing_device)
    print(device)

    # Caminho do arquivo CSV que você deseja ler
    csv_file_path = 'dados/frases.csv'

    # Crie uma lista vazia para armazenar as frases
    frases = []

    # Abra o arquivo CSV no modo leitura ('r') e use o objeto 'csv.reader' para ler o arquivo
    with open(csv_file_path, 'r', encoding='utf-8') as file:

        csv_reader = csv.reader(file)

        # Iterar sobre cada linha no arquivo CSV
        for row in csv_reader:
            frase = row[0]
            frases.append(frase)

    # Nome do objeto
    texts = frases

    # 1: positivo, 0: negativo
    labels = [1, 0, 1, 0, 1, 1, 0, 1, 1, 0]

    ''' Pré-Processamento '''

    RANDOM_SEED = 42

    # Divisão dos dados em treino e teste
    train_texts, test_texts, train_labels, test_labels = train_test_split(texts,
                                                                           labels,
                                                                           test_size=0.2,
                                                                           random_state=RANDOM_SEED)

    # Carrega o tokenizador
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    MAX_LENGTH = 64

    # Aplica a tokenização
    train_dataset = SentimentAnalysisTokenizer(train_texts, train_labels, tokenizer, MAX_LENGTH)

    # Aplica a tokenização
    test_dataset = SentimentAnalysisTokenizer(test_texts, test_labels, tokenizer, MAX_LENGTH)

    # Cria o data loader
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

    # Cria o data loader
    test_loader = DataLoader(test_dataset, batch_size=16)

    ''' Loop de Treino, Avaliação e Inferência '''

    # Função para treinar o modelo
    def train_epoch(model, data_loader, criterion, optimizer, device):

        model.train()
        total_loss = 0

        for batch in data_loader:

            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        return total_loss / len(data_loader)

    # Função para avaliar o modelo
    def eval_epoch(model, data_loader, criterion, device):

        model.eval()
        total_loss = 0

        with torch.no_grad():

            for batch in data_loader:

                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)

                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                total_loss += loss.item()

        return total_loss / len(data_loader)

    # Função para obter previsões do modelo
    def predict(model, data_loader, device):

        model.eval()
        predictions = []

        with torch.no_grad():

            for batch in data_loader:

                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)

                outputs = model(input_ids, attention_mask=attention_mask)
                _, preds = torch.max(outputs.logits, dim=1)
                predictions.extend(preds.tolist())

        return predictions

    ''' Construção do Modelo '''

    # Carregar o modelo
    modelo = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

    # Coloca o modelo na memória do device
    modelo.to(device)

    # Hiperparâmetros
    EPOCHS = 10
    LEARNING_RATE = 2e-5

    # Otimizador
    optimizer = torch.optim.AdamW(modelo.parameters(), lr=LEARNING_RATE)

    # Função de perda
    criterion = torch.nn.CrossEntropyLoss()

    ''' Treinamento e Avaliação do Modelo '''

    # Treinamento e validação do modelo
    for epoch in range(EPOCHS):
        train_loss = train_epoch(modelo, train_loader, criterion, optimizer, device)
        test_loss = eval_epoch(modelo, test_loader, criterion, device)
        print(f'Epoch {epoch+1}/{EPOCHS}, Train Loss: {train_loss}, Test Loss: {test_loss}')

    # Salva o modelo em disco
    torch.save(modelo, 'modelo.pt')

    ''' Testando o Modelo com Novos Dados '''

    # Novos dados
    novas_frases = ['Eu gostei muito deste filme.',
                    'O atendimento do restaurante foi decepcionante.']

    # Aplicando a tokenização
    dataset = SentimentAnalysisTokenizer(novas_frases, [0] * len(novas_frases), tokenizer, MAX_LENGTH)

    # Cria o data loader
    loader = DataLoader(dataset, batch_size=16)

    # Previsões
    previsoes = predict(modelo, loader, device)

    # Análise de sentimento
    for text, prediction in zip(novas_frases, previsoes):
        print(f'Sentença: {text} | Sentimento: {"positivo" if prediction else "negativo"}')
