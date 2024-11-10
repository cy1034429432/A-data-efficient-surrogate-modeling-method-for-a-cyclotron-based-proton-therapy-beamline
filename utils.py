"""
Author:Yu Chen
Email: yu_chen2000@hust.edu.cn
Personal website: hustyuchen.github.io
"""

import torch
import csv
from torch.utils.data import DataLoader, Dataset
import os
import pandas as pd
import numpy as np
import random
import torch.nn as nn
import torch.nn.functional as F

class BDSIM_from_Col2_to_PA_Dataset(Dataset):
    # this function is used to generate initial dataset and test dataset
    def __init__(self, dataset_address="./dataset", whether_is_training=True):
        super(BDSIM_from_Col2_to_PA_Dataset, self).__init__()
        self.whether_is_training = whether_is_training
        self.dataset_address = dataset_address
        self.input_data_label_list = ["Energy_label", "Q4E", "Q5E", "Q6E", "Q7E", "Q10E", "Q11E"]
        self.output_data_label_list = ["x_envelope_end", "y_envelope_end", "xp_envelope_end", "yp_envelope_end"]
        self.energy2wordembeding = {"70MeV": 0, "150MeV": 1, "230MeV": 2}
        self.Normal_list = {"Q4E": -5.4457, "Q5E": 5.0119, "Q6E": -1.4644, "Q7E": 2.7301, "Q10E": 4.484, "Q11E": -3.7974}

        if self.whether_is_training:
            self.root = str(self.dataset_address) + "/training_norm_dataset_5%.xlsx"
        else:
            self.root = str(self.dataset_address) + "/test_norm_dataset_5%.xlsx"
        self.data = np.array(pd.read_excel(self.root))
        self.data = self.data[:, 1:]
        self.data_number = self.data.shape[0]


    def load_initial_dataset_xlsx_and_create_dataset(self, orignal_dataset_adress="original_dataset"):

        total_data_list = []
        self.sample_number = 0

        # merge data
        xlsx_name_lists = os.listdir(orignal_dataset_adress)
        for xlsx_name in xlsx_name_lists:
            energy = xlsx_name[xlsx_name.find("PA_")+3:xlsx_name.find("MeV")+3]
            ratio = xlsx_name[xlsx_name.find("(")+1:xlsx_name.find("%")]
            label = self.energy2wordembeding[energy]
            data = pd.read_excel(os.path.join(orignal_dataset_adress, xlsx_name))
            data.insert(1, "Energy_label", label)
            self.sample_number += data.shape[0]
            total_data_list.append(data)
        total_data = pd.concat(total_data_list, ignore_index=True)

        ratio = float(ratio) * 0.01
        # random shuffle
        total_data = total_data.sample(frac=1)

        # clean data
        input_data_list = []
        output_data_list = []

        for input_data_label in self.input_data_label_list:
            if input_data_label == "Energy_label":
                processed_data = total_data[input_data_label]
            else:
                Normal_data = self.Normal_list[input_data_label]
                processed_data = (total_data[input_data_label] - Normal_data) / (abs(Normal_data) * ratio)
            input_data_list.append(processed_data)

        for output_data_label in self.output_data_label_list:
            output_data_list.append(total_data[output_data_label])

        input_data = pd.concat(input_data_list, axis=1, ignore_index=False)
        output_data = pd.concat(output_data_list, axis=1, ignore_index=False)

        # divide training dataset and test dataset
        training_number = int(self.sample_number*0.8)
        test_number = self.sample_number - training_number
        print(f"training number:{training_number}")
        print(f"test number:{test_number}")

        training_input_data = input_data[0:training_number]
        training_output_data = output_data[0:training_number]

        test_input_data = input_data[training_number:]
        test_output_data = output_data[training_number:]

        training_dataset = pd.concat((training_input_data, training_output_data), axis=1, ignore_index=False)
        test_dataset = pd.concat((test_input_data, test_output_data), axis=1, ignore_index=False)



        training_dataset.to_excel(f"./dataset/training_norm_dataset_{int(ratio*100)}%.xlsx")
        test_dataset.to_excel(f"./dataset/test_norm_dataset_{int(ratio*100)}%.xlsx")

    def __len__(self):
        return self.data_number

    def __getitem__(self, item):
        data = self.data[item]
        data = torch.tensor(data, dtype=torch.float32)
        return (data[0:7], data[7:11])

class Select_training_dataset(Dataset):
    def __init__(self, selected_training_dataset_csv, training_dataset_address="./dataset/training_norm_dataset_5%.xlsx"):
        super(Select_training_dataset, self).__init__()
        self.data = np.array(pd.read_excel(training_dataset_address))
        self.data = self.data[:, 1:]
        self.training_dataset_number_list = self.load_csv(selected_training_dataset_csv)


    def load_csv(self, filename):
        sample_list = []
        with open (filename, mode="r", newline="") as f:
            reader = csv.reader(f)
            for row in reader:
                sample_list.append(int(row[0]))
        return sample_list


    def __len__(self):
        return len(self.training_dataset_number_list)

    def __getitem__(self, item):
        item = self.training_dataset_number_list[item]
        data = self.data[item]
        data = torch.tensor(data, dtype=torch.float32)
        return (data[0:7], data[7:], item)



class Teacher_training_dataset(Dataset):
    def __init__(self, selected_training_dataset_csv, training_dataset_address="./dataset/training_norm_dataset_5%.xlsx"):
        super(Teacher_training_dataset, self).__init__()
        self.data = np.array(pd.read_excel(training_dataset_address))
        self.data = self.data[:, 1:]
        self.sample_list, self.label_list = self.load_csv(filename=selected_training_dataset_csv)
        self.Normal_list = [-5.4457, 5.0119, -1.4644, 2.7301, 4.484, -3.7974]

    def load_csv(self, filename):
        sample_list, label_list = [], []
        with open(filename, mode="r", newline="") as f:
            reader = csv.reader(f)
            for row in reader:
                sample, label = row
                label = int(label)
                sample_list.append(sample)
                label_list.append(label)

        assert len(sample_list) == len(label_list)
        return sample_list, label_list


    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, item):
        sampel, label = self.sample_list[item], self.label_list[item]
        y = torch.tensor(label)
        data = self.data[int(sampel), :]
        x = torch.tensor(data[0:7], dtype=torch.float32)
        return x, y



def write_initial_dataset(initial_training_filename, initial_unselected_filename, sample_number=512):

    if os.path.exists(initial_training_filename):
        os.remove(initial_training_filename)
    if os.path.exists(initial_unselected_filename):
        os.remove(initial_unselected_filename)

    sequency = range(0, int(27000 * 0.8))
    number_list = random.sample(sequency, sample_number)
    unselected_sample_list = [a for a in sequency]

    for number in number_list:
        unselected_sample_list.remove(number)

    with open(initial_training_filename, mode="w", newline="") as f:
        writer = csv.writer(f)
        for number in number_list:
            writer.writerow([number])

    with open(initial_unselected_filename, mode="w", newline="") as f:
        writer = csv.writer(f)
        for number in unselected_sample_list:
            writer.writerow([number])


def wirte_teacher_dataset_csv(sample_list, label_list, filename):

    if os.path.exists(filename):
        os.remove(filename)
    for number in range(len(sample_list)):
        with open(filename, mode="a", newline="") as f:
            writer = csv.writer(f)
            for sub_number in range(len(sample_list[number])):
                writer.writerow([sample_list[number][sub_number], label_list[number][sub_number]])

class Student_Representation_learning(nn.Module):
    def __init__(self, input_size=6, output_size=4, feature_size=16, embed_size=30, energy_classes=3, whether_bidirectional=True):
        super(Student_Representation_learning, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.embed_size = embed_size
        self.energy_classes = energy_classes
        self.whether_bidirectional = whether_bidirectional
        self.feature_size = feature_size

        #self.rnn = nn.LSTM(input_size=self.input_size, hidden_size=self.feature_size, num_layers=1, bidirectional=self.whether_bidirectional, dropout=0.5)
        self.rnn = nn.Linear(self.input_size, self.feature_size * 2)

        self.embed = nn.Embedding(num_embeddings=self.energy_classes, embedding_dim=self.embed_size)
        nn_feature = feature_size * 2 if self.whether_bidirectional == True else feature_size
        self.active_function = nn.ReLU()
        self.last_active_function = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(nn_feature + self.embed_size, self.feature_size * 4)
        self.fc2 = nn.Linear(self.feature_size * 4, self.feature_size * 2)
        self.fc3 = nn.Linear(self.feature_size * 2, self.output_size)


    def get_label_for_teacher_network(self, real_label, predict_label, bound_rate=0.05):
        real_label = real_label.detach().cpu().numpy()
        predict_label = predict_label.detach().cpu().numpy()
        realtive_error = np.sum(abs(real_label - predict_label) / abs(real_label), axis=1) / 4
        label = np.where(realtive_error < bound_rate, 1, 0)
        label = label.tolist()
        return label


    def forward(self, input):
        #  data conversation
        energy_label = input[:, 0].long()
        Q_strength = input[:, 1:].float()
        embedding = self.embed(energy_label).view(input.shape[0], -1)
        output = self.rnn(Q_strength)
        output = torch.cat([embedding, output], dim=1)
        output = self.dropout(self.active_function(self.fc1(output)))
        last_layer_output = self.dropout(self.active_function(self.fc2(output)))
        output = self.last_active_function(self.fc3(last_layer_output))
        return (output, last_layer_output)




class CNN_linear(nn.Module):
    def __init__(self, input_size=6, hidden_size=512, feature_size=16, embed_size=30, energy_classes=3):
        super(CNN_linear, self).__init__()
        self.CNN1 = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=3)
        self.CNN2 = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=3)
        self.CNN3 = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=3)
        self.fc1 = nn.Linear(30, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size * 2)
        self.fc3 = nn.Linear(hidden_size * 2, hidden_size)
        self.fc4 = nn.Linear(hidden_size, 4)
        self.embed = nn.Embedding(energy_classes, embed_size)
        self.activefunction = nn.ReLU()
        self.BN1 = nn.BatchNorm1d(hidden_size)
        self.BN2 = nn.BatchNorm1d(hidden_size * 2)
        self.BN3 = nn.BatchNorm1d(hidden_size)

    def forward(self, input):
        energy_label = input[:, 0].long()
        Q_strength = input[:, 1:].float()
        embedding = self.embed(energy_label).view(input.shape[0], -1)
        output = torch.cat([embedding, Q_strength], dim=1).reshape(input.shape[0], 1, -1)
        output = self.activefunction(self.CNN1(output))
        output = self.activefunction(self.CNN2(output))
        output = self.activefunction(self.CNN3(output))
        output = output.reshape(input.shape[0], -1)
        output = self.activefunction(self.fc1(output))
        output = self.BN1(output)
        output = self.activefunction(self.fc2(output))
        output = self.BN2(output)
        output = self.activefunction(self.fc3(output))
        output = self.BN3(output)
        output = self.activefunction(self.fc4(output))
        return output




class GRU_linear(nn.Module):
    def __init__(self, input_size=6, hidden_size=512,  embed_size=6, energy_classes=3):
        super(GRU_linear, self).__init__()
        rnn_size = 15
        self.GRU = nn.GRU(input_size=1, hidden_size=rnn_size, num_layers=1)
        #self.RNN = nn.LSTM(input_size=1, hidden_size=rnn_size, num_layers=3)
        self.fc1 = nn.Linear((embed_size+input_size) * rnn_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size * 2)
        self.fc3 = nn.Linear(hidden_size * 2, hidden_size)
        self.fc4 = nn.Linear(hidden_size, 4)
        self.embed = nn.Embedding(energy_classes, embed_size)
        self.Drop = nn.Dropout()
        self.BN1 = nn.BatchNorm1d(hidden_size)
        self.BN2 = nn.BatchNorm1d(hidden_size * 2)
        self.BN3 = nn.BatchNorm1d(hidden_size)
        self.activefunction = nn.ReLU()

    def forward(self, input):
        energy_label = input[:, 0].long()
        Q_strength = input[:, 1:].float()
        embedding = self.embed(energy_label).view(input.shape[0], -1)
        output = torch.cat([embedding, Q_strength], dim=1).reshape(input.shape[0], -1, 1)

        output, _ = self.GRU(output)
        output = output.reshape(input.shape[0], -1)
        output = self.activefunction(self.fc1(output))
        output = self.BN1(output)
        output = self.activefunction(self.fc2(output))
        output = self.BN2(output)
        output = self.activefunction(self.fc3(output))
        output = self.BN3(output)
        output = self.activefunction(self.fc4(output))
        return output


class LSTM_linear(nn.Module):
    def __init__(self, input_size=6, hidden_size=512,  embed_size=6, energy_classes=3):
        super(LSTM_linear, self).__init__()
        rnn_size = 15
        self.LSTM = nn.LSTM(input_size=1, hidden_size=rnn_size, num_layers=1)
        #self.RNN = nn.LSTM(input_size=1, hidden_size=rnn_size, num_layers=3)
        self.fc1 = nn.Linear((embed_size+input_size) * rnn_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size * 2)
        self.fc3 = nn.Linear(hidden_size * 2, hidden_size)
        self.fc4 = nn.Linear(hidden_size, 4)
        self.embed = nn.Embedding(energy_classes, embed_size)
        self.Drop = nn.Dropout()
        self.BN1 = nn.BatchNorm1d(hidden_size)
        self.BN2 = nn.BatchNorm1d(hidden_size * 2)
        self.BN3 = nn.BatchNorm1d(hidden_size)
        self.activefunction = nn.ReLU()

    def forward(self, input):
        energy_label = input[:, 0].long()
        Q_strength = input[:, 1:].float()
        embedding = self.embed(energy_label).view(input.shape[0], -1)
        output = torch.cat([embedding, Q_strength], dim=1).reshape(input.shape[0], -1, 1)

        output, (_, _) = self.LSTM(output)
        output = output.reshape(input.shape[0], -1)
        output = self.activefunction(self.fc1(output))
        output = self.BN1(output)
        output = self.activefunction(self.fc2(output))
        output = self.BN2(output)
        output = self.activefunction(self.fc3(output))
        output = self.BN3(output)
        output = self.activefunction(self.fc4(output))
        return output





class MLP(nn.Module):
    def __init__(self, input_size=6, hidden_size=512, embed_size=6, energy_classes=3):
        super(MLP, self).__init__()

        self.emd = nn.Embedding(num_embeddings=energy_classes, embedding_dim=embed_size)

        self.fc1 = nn.Linear(input_size + embed_size, hidden_size)

        #self.fc1 = nn.Linear(7, hidden_size)

        self.fc2 = nn.Linear(hidden_size, hidden_size * 2)
        self.fc3 = nn.Linear(hidden_size * 2, hidden_size)
        self.fc4 = nn.Linear(hidden_size, 4)

        self.BN1 = nn.BatchNorm1d(hidden_size)
        self.BN2 = nn.BatchNorm1d(hidden_size * 2)
        self.BN3 = nn.BatchNorm1d(hidden_size)
        self.activefunction = nn.ReLU()

    def get_label_for_teacher_network(self, real_label, predict_label, bound_rate=0.05):
        real_label = real_label.detach().cpu().numpy()
        predict_label = predict_label.detach().cpu().numpy()
        realtive_error = np.sum(abs(real_label - predict_label) / abs(real_label), axis=1) / 4
        label = np.where(realtive_error < bound_rate, 1, 0)
        label = label.tolist()
        return label

    def forward(self, input):
        label = input[:, 0].long()
        label_fea = self.emd(label).view(input.shape[0], -1)
        input = torch.cat([input[:, 1:], label_fea], dim=1)
        output = self.activefunction(self.fc1(input))
        output = self.BN1(output)
        output = self.activefunction(self.fc2(output))
        output = self.BN2(output)
        output = self.activefunction(self.fc3(output))
        output = self.BN3(output)
        output = self.activefunction(self.fc4(output))
        return output



class Teacher_Representation_learning(nn.Module):
    def __init__(self, input_size=6, output_size=2, feature_size=16, embed_size=30, energy_classes=3, whether_bidirectional=False):
        super(Teacher_Representation_learning, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.feature_size = feature_size
        self.embed_size = embed_size
        self.energy_classes = energy_classes
        self.whether_bidirectional = whether_bidirectional
        self.embed = nn.Embedding(num_embeddings=self.energy_classes, embedding_dim=self.embed_size)
        self.cnn_1 = self._block(in_channels=1, out_channels=feature_size * 2, kernel_size=3, stride=1)
        self.cnn_2 = self._block(in_channels=feature_size * 2, out_channels=feature_size * 4, kernel_size=6, stride=1)
        self.cnn_3 = self._block(in_channels=feature_size * 4, out_channels=1, kernel_size=6, stride=1)
        self.fc_1 = nn.Linear(in_features=18 + self.embed_size, out_features=feature_size * 4)
        self.fc_2 = nn.Linear(in_features=feature_size * 4, out_features=feature_size * 2)
        self.fc_3 = nn.Linear(in_features=feature_size * 2, out_features=feature_size)
        self.fc_4 = nn.Linear(in_features=feature_size, out_features=self.output_size)
        self.active_function = nn.ReLU()

    def _block(self, in_channels, out_channels, kernel_size, stride):
        return nn.Sequential(
            nn.ConvTranspose1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride),
            nn.BatchNorm1d(out_channels),
            nn.ReLU())

    def forward(self, input):
        energy_label = input[:, 0].long()
        Q_strength = input[:, 1:].float().view(input.shape[0], 1, -1)
        embedding = self.embed(energy_label).view(input.shape[0], -1)
        output = self.cnn_1(Q_strength)
        output = self.cnn_2(output)
        output = self.cnn_3(output).view(input.shape[0], -1)
        output = torch.cat([embedding, output], dim=1)
        output = self.active_function(self.fc_1(output))
        output = self.active_function(self.fc_2(output))
        last_two_layer = self.active_function(self.fc_3(output))
        last_layer = torch.sigmoid(self.fc_4(last_two_layer))
        return (last_layer, last_two_layer)


class TNet(nn.Module):
    def __init__(self, input_size=7, hidden_size=512, output_size=2):
        super(TNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size * 2)
        self.fc3 = nn.Linear(hidden_size * 2, hidden_size)
        self.fc4 = nn.Linear(hidden_size, output_size)

        self.BN1 = nn.BatchNorm1d(hidden_size)
        self.BN2 = nn.BatchNorm1d(hidden_size * 2)
        self.BN3 = nn.BatchNorm1d(hidden_size)
        self.active_function = nn.ReLU()
        self.active_function_2 = nn.Sigmoid()

    def forward(self, x):
        x = self.active_function(self.fc1(x))
        x = self.BN1(x)
        x = self.active_function(self.fc2(x))
        x = self.BN2(x)
        last_two_layer = self.active_function(self.fc3(x))
        x = self.BN3(last_two_layer)
        last_layer = self.active_function(self.fc4(x))
        return (last_layer, last_two_layer)




class MLP_teacher(nn.Module):
    def __init__(self, input_size=6, hidden_size=512, embed_size=6, energy_classes=3):
        super(MLP_teacher, self).__init__()

        self.emd = nn.Embedding(num_embeddings=energy_classes, embedding_dim=embed_size)

        self.fc1 = nn.Linear(input_size + embed_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size * 2)
        self.fc3 = nn.Linear(hidden_size * 2, 2)


        self.BN1 = nn.BatchNorm1d(hidden_size)
        self.BN2 = nn.BatchNorm1d(hidden_size * 2)
        self.BN3 = nn.BatchNorm1d(hidden_size)
        self.activefunction = nn.ReLU()

    def get_label_for_teacher_network(self, real_label, predict_label, bound_rate=0.05):
        real_label = real_label.detach().cpu().numpy()
        predict_label = predict_label.detach().cpu().numpy()
        realtive_error = np.sum(abs(real_label - predict_label) / abs(real_label), axis=1) / 4
        label = np.where(realtive_error < bound_rate, 1, 0)
        label = label.tolist()
        return label

    def forward(self, input):
        label = input[:, 0].long()
        label_fea = self.emd(label).view(input.shape[0], -1)
        input = torch.cat([input[:, 1:], label_fea], dim=1)
        output = self.activefunction(self.fc1(input))
        output = self.BN1(output)
        output = self.activefunction(self.fc2(output))
        output = self.BN2(output)
        output = self.activefunction(self.fc3(output))
        return output, output




def initialize_weight(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d, nn.BatchNorm1d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight.data)

if __name__ == '__main__':
    a = BDSIM_from_Col2_to_PA_Dataset(Dataset)
    a.load_initial_dataset_xlsx_and_create_dataset()


