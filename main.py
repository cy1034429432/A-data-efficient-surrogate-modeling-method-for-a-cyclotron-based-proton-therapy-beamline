"""
Author:Yu Chen
Email: yu_chen2000@hust.edu.cn
Personal website: hustyuchen.github.io
"""


import csv
import os
import pandas as pd
import torch
import numpy as np
import random
from active_learning_utils import *
from utils import *
import datetime
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
from torch import optim


def initial_setup_and_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return device


def initial_model_set(device):
    model_student = MLP()
    model_teacher = TNet()
    initialize_weight(model_student)
    initialize_weight(model_teacher)
    model_student.to(device)
    model_teacher.to(device)
    student_loss_function = nn.MSELoss()
    teacher_loss_function = nn.CrossEntropyLoss()
    optimizer_model_student = optim.Adam(model_student.parameters(), lr=0.0001, betas=(0.0, 0.9))
    optimizer_model_teacher = optim.Adam(model_teacher.parameters(), lr=0.0001, betas=(0.0, 0.9))

    return (model_student, model_teacher, student_loss_function, teacher_loss_function,
            optimizer_model_student, optimizer_model_teacher)


def dataset_prepare(sample_number=2000):
    write_initial_dataset("./dataset/initial_student_training_set.csv",
                          "./dataset/initial_unselected_training_set.csv",
                          sample_number=sample_number)
    student_test_dataset = BDSIM_from_Col2_to_PA_Dataset(whether_is_training=False)
    student_test_dataset_loader = DataLoader(student_test_dataset, batch_size=512, num_workers=0)
    return student_test_dataset_loader

def training_student_model(device, model_student, student_dataset_csv, student_test_dataset_loader, student_loss_function,
                           student_training_epoch, optimizer_model_student, write_name, model_save_address, whether_is_initial=False):


    def get_accuracy(real_label, predict_label, bound_rate=0.05):
        real_label = real_label.detach().cpu().numpy()
        predict_label = predict_label.detach().cpu().numpy()
        realtive_error = np.sum(abs(real_label - predict_label) / abs(real_label), axis=1) / 4
        label = np.where(realtive_error < bound_rate, 1, 0)
        number = label.sum()
        return number



    if whether_is_initial == False:
        student_dataset = Select_training_dataset(student_dataset_csv)
    else:
        student_dataset = Select_training_dataset("./dataset/initial_student_training_set.csv")

    student_data_loader = DataLoader(student_dataset, batch_size=32, num_workers=0)
    total_bk_step = 0
    write = SummaryWriter(write_name)
    test_loss_function = nn.MSELoss(reduction="sum")
    MSE_list = [100]
    Accuracy_5_list = []
    Accuracy_3_list = []


    if os.path.exists(f"{model_save_address}") == False:
        os.makedirs(model_save_address)

    for epoch in range(student_training_epoch):
        model_student.train()
        for number, (student_input, student_label, item) in enumerate(student_data_loader):
            student_input, student_label= student_input.to(device), student_label.to(device)
            model_student_output = model_student(student_input)
            loss = student_loss_function(model_student_output, student_label)

            loss = loss.float()
            L1_reg = 0
            for param in model_student.parameters():
                L1_reg += torch.sum(torch.abs(param))
            loss = loss + 0.0001 * L1_reg

            optimizer_model_student.zero_grad()
            loss.backward()
            optimizer_model_student.step()
            write.add_scalar("training_loss", loss, total_bk_step)
            total_bk_step += 1

        total_loss = 0
        test_number = 0

        total_accuracy_5 = 0
        total_accuracy_3 = 0

        model_student.eval()
        with torch.no_grad():
            for _, (input, label) in enumerate(student_test_dataset_loader):
                input, label = input.to(device), label.to(device)
                output = model_student(input)
                loss = test_loss_function(label, output)
                total_loss += loss
                test_number += input.shape[0]

                accuracy_5 = get_accuracy(label, output, bound_rate=0.05)
                accuracy_3 = get_accuracy(label, output, bound_rate=0.03)
                total_accuracy_5 = accuracy_5 + total_accuracy_5
                total_accuracy_3 = accuracy_3 + total_accuracy_3


            MSE = total_loss/test_number
            total_accuracy_5 = total_accuracy_5 / test_number
            total_accuracy_3 = total_accuracy_3 / test_number

            print(f"training epochs:{epoch+1}/{student_training_epoch}, test MSE:{MSE}")
            write.add_scalar("training_loss", MSE, epoch)
            if MSE < min(MSE_list):
                best_student_model_parameter_address = f"{model_save_address}/student_model_MSE_{MSE}.pt"
                torch.save(model_student.state_dict(), f"{model_save_address}/student_model_MSE_{MSE}.pt")
            MSE_list.append(MSE)
            Accuracy_5_list.append(total_accuracy_5)
            Accuracy_3_list.append(total_accuracy_3)

    best_MSE = min(MSE_list)
    best_Accuracy_5 = max(Accuracy_5_list)
    best_Accuracy_3 = max(Accuracy_3_list)
    write.close()

    return (best_student_model_parameter_address, best_MSE, best_Accuracy_5, best_Accuracy_3, student_data_loader)

def label_teacher_dataset(device, model_student, student_data_loader, best_student_model_parameter_address,
                          teacher_dataset_name):

    model_student.load_state_dict(torch.load(best_student_model_parameter_address))
    model_student.eval()
    item_list = []
    label_list = []
    with torch.no_grad():
        for number, (student_input, student_label, item) in enumerate(student_data_loader):
            student_input, student_label = student_input.to(device), student_label.to(device)
            model_student_output= model_student(student_input)
            label = model_student.get_label_for_teacher_network(student_label, model_student_output, bound_rate=0.05)
            label_list.append(label)
            item_list.append(item.detach().numpy().tolist())

    wirte_teacher_dataset_csv(item_list, label_list, f"{teacher_dataset_name}")
    teacher_dataset = Teacher_training_dataset(f"{teacher_dataset_name}")
    teacher_dataset_loader = DataLoader(teacher_dataset, batch_size=32, num_workers=0)
    return teacher_dataset_loader


def training_teacher_model(device, model_teacher, teacher_loss_function, optimizer_model_teacher,
                           teacher_dataset_loader, teacher_training_epoch, write_name, model_save_address):

    if os.path.exists(model_save_address) == False:
        os.makedirs(model_save_address)

    write = SummaryWriter(f"{write_name}")
    model_teacher.to(device)

    training_epoches = teacher_training_epoch
    total_back = 0
    Accuracy_list = [0]

    for epoch in range(training_epoches):
        model_teacher.train()

        for (teacher_input, teacher_output) in teacher_dataset_loader:
            teacher_input, teacher_output = teacher_input.to(device), teacher_output.to(device)
            model_output, _ = model_teacher(teacher_input)
            loss = teacher_loss_function(model_output, teacher_output)
            optimizer_model_teacher.zero_grad()
            loss.backward()
            optimizer_model_teacher.step()
            write.add_scalar("training_loss", loss, total_back)
            total_back += 1

        model_teacher.eval()
        total_accuracy = 0
        total_number = 0
        with torch.no_grad():
            for (teacher_input, teacher_output) in teacher_dataset_loader:
                teacher_input, teacher_output = teacher_input.to(device), teacher_output.to(device)
                (model_output, _) = model_teacher(teacher_input)

                accuracy = (model_output.argmax(1)==teacher_output).sum()
                total_number += teacher_input.shape[0]
                total_accuracy += accuracy

            accuracy = total_accuracy / total_number

            write.add_scalar("teacher_accuracy", accuracy, epoch)
            print(f"teacher network, epoch:{epoch + 1}/{training_epoches}, accuracy:{accuracy}")

            if accuracy > max(Accuracy_list):
                best_teacher_model_parameter_address = f"{model_save_address}/teacher_model_Accuracy_{accuracy}.pt"
                torch.save(model_teacher.state_dict(), f"{model_save_address}/teacher_model_Accuracy_{accuracy}.pt")

    write.close()
    return best_teacher_model_parameter_address


def use_teacher_model_combined_with_active_learning_to_select_sample(device, model_teacher, best_teacher_model_parameter_address,
                                                                     active_learning_strategy, sample_number,
                                                                     pool_dataset_csv_address, selected_dataset_csv_address,
                                                                     new_student_selected_address, new_student_unselected_address,
                                                                     whether_is_initial=False):
    if whether_is_initial == False:
        pool_dataset = Select_training_dataset(pool_dataset_csv_address)
    else:
        pool_dataset_csv_address = "./dataset/initial_unselected_training_set.csv"
        selected_dataset_csv_address = "./dataset/initial_student_training_set.csv"
        pool_dataset = Select_training_dataset("./dataset/initial_unselected_training_set.csv")


    pool_dataset_loader = DataLoader(pool_dataset, batch_size=100, num_workers=0)

    model_outputs = pd.DataFrame()
    model_last_layer_outputs = pd.DataFrame()
    pool_items = pd.DataFrame()

    model_teacher.load_state_dict(torch.load(best_teacher_model_parameter_address))
    model_teacher.to(device)


    model_teacher.eval()
    with torch.no_grad():
        for (pool_input, _, pool_item) in pool_dataset_loader:
            pool_input = pool_input.to(device)
            model_output, model_last_layer_output = model_teacher(pool_input)
            model_output = pd.DataFrame(model_output.cpu().detach().numpy())
            model_last_layer_output = pd.DataFrame(model_last_layer_output.cpu().detach().numpy())
            pool_item = pd.DataFrame(pool_item.cpu().detach().numpy())

            model_outputs = model_outputs._append(model_output, ignore_index=True)
            model_last_layer_outputs = model_last_layer_outputs._append(model_last_layer_output, ignore_index=True)
            pool_items = pool_items._append(pool_item, ignore_index=True)

    model_outputs = np.array(model_outputs)
    model_last_layer_outputs = np.array(model_last_layer_outputs)
    pool = np.array(pool_items).tolist()
    pool_items = []
    for number in pool:
        pool_items.append(int(number[0]))

    if (active_learning_strategy.__name__ == "query_max_uncertainty") or (active_learning_strategy.__name__ == "query_margin_prob") \
        or (active_learning_strategy.__name__ == "query_max_entropy") or (active_learning_strategy.__name__ == "random_sampleing"):
        new_selected_sample_list = active_learning_strategy(model_outputs, sample_number)
    elif active_learning_strategy.__name__ == "query_margin_kmeans":
        new_selected_sample_list = active_learning_strategy(model_outputs, model_last_layer_outputs, sample_number)
    elif active_learning_strategy.__name__ == "query_margin_kmeans_pure_diversity":
        new_selected_sample_list = active_learning_strategy(model_last_layer_outputs, sample_number)
    elif active_learning_strategy.__name__ == "query_margin_kmeans_2stages":
        # B : We suggest B choose 3~5
        new_selected_sample_list = active_learning_strategy(model_outputs, model_last_layer_outputs, sample_number, 10)

    unselected_sample_list = []
    selected_sample_list = []

    with open(pool_dataset_csv_address, mode="r", newline="") as f:
        reader = csv.reader(f)
        reader = list(reader)
        for sample in reader:
            unselected_sample_list.append(int(sample[0]))

    with open(selected_dataset_csv_address, mode="r", newline="") as f:
        reader = csv.reader(f)
        reader = list(reader)
        for sample in reader:
            selected_sample_list.append(int(sample[0]))


    for new_selected_sample in new_selected_sample_list:
        unselected_sample_list.remove(pool_items[new_selected_sample])
        selected_sample_list.append(pool_items[new_selected_sample])


    # write new sample csv
    with open(f"{new_student_selected_address}", mode="w", newline="") as f:
        write = csv.writer(f)
        for sample in selected_sample_list:
            write.writerow([sample])

    with open(f"{new_student_unselected_address}", mode="w", newline="") as f:
        write = csv.writer(f)
        for sample in unselected_sample_list:
            write.writerow([sample])



if __name__ == '__main__':

    device = initial_setup_and_seed(3403)
    active_learning_step = 20
    student_training_epoch = 50
    teacher_training_epoch = 50
    sample_number = 128
    begin_sample = 300
    # inintial model setting and prepare dataset

    (model_student, model_teacher, student_loss_function, teacher_loss_function,
     optimizer_model_student, optimizer_model_teacher) = initial_model_set(device)

    student_test_dataset_loader = dataset_prepare(sample_number=begin_sample)

    #active_learning_strategy_list = [query_max_uncertainty, query_max_entropy, query_margin_kmeans, query_margin_kmeans_pure_diversity, query_margin_kmeans_2stages]


    # random sampling
    active_learning_strategy_list = [query_max_entropy]


    #training intial dataset by random sampling
    for active_learning_strategy in active_learning_strategy_list:

        write = SummaryWriter(f"./log/{active_learning_strategy.__name__}")
        for number in range(active_learning_step):

            if number == 0:
                whether_is_initial = True
            else:
                whether_is_initial = False

            best_student_model_parameter_address,  best_MSE, best_Accuracy_5, best_Accuracy_3, student_data_loader = training_student_model(
                                                    device=device, model_student=model_student,
                                                    student_dataset_csv=f"./dataset/training_student_dataset_step_{number}.csv",
                                                    student_test_dataset_loader=student_test_dataset_loader,
                                                    student_loss_function=student_loss_function,
                                                    student_training_epoch=student_training_epoch,
                                                    optimizer_model_student=optimizer_model_student,
                                                    write_name=f"./log/{active_learning_strategy.__name__}/student/strategy_step_{number}",
                                                    model_save_address=f"./model_parameter/{active_learning_strategy.__name__}/student/strategy_step_{number}/",
                                                    whether_is_initial=whether_is_initial)

            write.add_scalar("test_MSE", best_MSE, begin_sample + (number) * sample_number)
            write.add_scalar("best_Accuracy_5", best_Accuracy_5, begin_sample + (number) * sample_number)
            write.add_scalar("best_Accuracy_3", best_Accuracy_3, begin_sample + (number) * sample_number)

            teacher_dataset_loader = label_teacher_dataset(device=device, model_student=model_student,
                                                    student_data_loader=student_data_loader,
                                                    best_student_model_parameter_address=best_student_model_parameter_address,
                                                    teacher_dataset_name=f"./dataset/training_teacher_dataset_step_{number}.csv")

            best_teacher_model_parameter_address = training_teacher_model(device=device, model_teacher=model_teacher,
                                                    teacher_loss_function=teacher_loss_function, optimizer_model_teacher=optimizer_model_teacher,
                                                    teacher_dataset_loader=teacher_dataset_loader,
                                                    teacher_training_epoch=teacher_training_epoch,
                                                    write_name=f"./log/{active_learning_strategy.__name__}/teacher/strategy_step_{number}",
                                                    model_save_address=f"./model_parameter/{active_learning_strategy.__name__}/teacher/strategy_step_{number}/")
            #
            #
            use_teacher_model_combined_with_active_learning_to_select_sample(device=device, model_teacher=model_teacher,
                                                   best_teacher_model_parameter_address=best_teacher_model_parameter_address,
                                                   active_learning_strategy=active_learning_strategy,
                                                   sample_number=sample_number,
                                                   pool_dataset_csv_address=f"./dataset/unselected_dataset_{number}.csv",
                                                   selected_dataset_csv_address=f"./dataset/training_student_dataset_step_{number}.csv",
                                                   new_student_selected_address=f"./dataset/training_student_dataset_step_{number+1}.csv",
                                                   new_student_unselected_address=f"./dataset/unselected_dataset_{number+1}.csv",
                                                   whether_is_initial=whether_is_initial)

        write.close()


