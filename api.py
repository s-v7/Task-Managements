import time
import torch
import string
import sqlite3
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
#from plyer import notification


#Neural Network Model Definition in PyTorch
class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralNetwork, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        return x

def train_neural_network(model, inputs, targets, epochs, lr=0.001): 
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    losses = []  # Para armazenar a loss em cada época

    for epoch in range(epochs):
        output = model(inputs)
        loss = criterion(output, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        if epoch % 10 == 0:  # Imprimir a loss a cada 10 épocas
            print(f'Época {epoch}, Loss: {loss.item()}')
    return model

def evaluate_model(model, test_inputs, test_targets):
  criterion = nn.MSELoss()
  with torch.no_grad():
    test_output = model(test_inputs)
    test_loss = criterion(test_output, test_targets)
    print(f'Loss in test data: {test_loss.item()}')

class PyTorchWrapper(BaseEstimator, RegressorMixin):
    def __init__(self, model, criterion, optimizer, epochs=100, lr=0.001):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.epochs = epochs
        self.lr = lr

    def fit(self, X, y):
        X, y = check_X_y(X, y, accept_sparse=True)

        inputs = torch.tensor(X, dtype=torch.float32)
        targets = torch.tensor(y, dtype=torch.float32).view(-1, 1)

        for epoch in range(self.epochs):
            output = self.model(inputs)
            loss = self.criterion(output, targets)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        return self

    def predict(self, X):
        check_is_fitted(self)
        X = check_array(X, accept_sparse=True)
        inputs = torch.tensor(X, dtype=torch.float32)
        with torch.no_grad():
            predictions = self.model(inputs)
        return predictions.numpy()


# Custom metric function
def accuracy_metric(y_true, y_pred):
    # Converting the predictions to the range [0, 10]
    y_pred = np.clip(y_pred, 0, 10)
    
    # Rounding to the nearest integer
    y_pred_rounded = np.round(y_pred)
    
    # Calculating accuracy as the average of correct predictions
    accuracy = np.mean(y_pred_rounded == y_true)
    
    return accuracy

# Creating a scorer using the custom metric function
accuracy_scorer = make_scorer(accuracy_metric, greater_is_better=True)

# Function to predict priorities based on tasks
def predict_priority(model, input_data):
    with torch.no_grad():
        return model(input_data).item()

# Function to create SQLite database
def create_database():
    connection = sqlite3.connect('tasks.db')
    cursor = connection.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS tasks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            description TEXT NOT NULL,
            priority REAL NOT NULL,
            due_date TEXT 
        )
    ''')
    connection.commit()
    connection.close()

# Run the create_database function again
create_database()
# Function to insert a new task into the database
def insert_task(description, priority, due_date):
    try:
        # Try to insert the task into the database
        connection = sqlite3.connect('tasks.db')
        cursor = connection.cursor()
        cursor.execute('INSERT INTO tasks (description, priority, due_date) VALUES (?, ?, ?)', (description, priority, due_date))
        connection.commit()
        connection.close()
    except (ValueError, sqlite3.Error) as e:
        print(f"SQLite error: {e}")
        print(f"Some priorities are not valid numbers. in: {cursor}")


def clear_last_rows(num_rows_to_clear=8):
    connection = sqlite3.connect('tasks.db')
    cursor = connection.cursor()

    # Get the total number of rows in the table
    cursor.execute('SELECT COUNT(*) FROM tasks')
    total_rows = cursor.fetchone()[0]

    # Calculate the index from which the last rows will be removed
    start_index = max(0, total_rows - num_rows_to_clear)

    # Remove last lines
    cursor.execute(f'DELETE FROM tasks WHERE id >= {start_index}')

    connection.commit()
    connection.close()

# Function to get all database tasks
def get_all_tasks():
 try:
    connection = sqlite3.connect('tasks.db')
    cursor = connection.cursor()
    cursor.execute('SELECT id, description, priority, due_date FROM tasks WHERE priority IS NOT NULL')
    tasks = cursor.fetchall()
    connection.close()
    return tasks
 except sqlite3.Error as e:
    print(f"SQlite error: {e}")

def get_user_priority():
    while True:
        try:
            priority = input("Enter the Task priority(0-10): ")

            # Validate if input is a number
            if not priority.replace(".", "").isdigit():
                print("Please enter a valid priority number.")
                continue

            priority = float(priority)

            if 0 <= priority <= 10:
                return priority
            else:
                print("Priority must be between 0 and 10.")
        except ValueError:
                print(f"Error: Some priorities are not valid numbers.: {get_user_priority()}")
                return None               
        
def get_user_choice():
    while True:
        choice = input("Choose an option(1, 2, 3, 4, 5, 6): ")
        if choice in {'1', '2', '3', '4', '5', '6'}:
            return choice
        else:
            print("Invalid choice. Type it 1, 2, 3, 4, 5 or 6")

def display_menu():
    print("\n======= Menu ==========")
    print("1. Add new task      ||")
    print("2. View tasks        ||")
    print("3. Train the model   ||")
    print("4. Schedule task     ||")
    print("5. Check Schedules   ||")
    print("6. To go out         ||")
    print("======== Menu =========")
def process_description(description): 
  # Convert to lowercase
  descriptions = description.lower()

  # Remove scores
  descriptions = descriptions.translate(str.maketrans("", "", string.punctuation))

  return descriptions

# Add the ability to save and load the model so previous training isn't lost:
def save_model(model, path="model.pth"):
    torch.save(model.state_dict(), path)
    print(f"Model saved in {path}")

def load_model(model, path="model.pth"):
    try:
        model.load_state_dict(torch.load(path))
        print(f"Model loaded with {path}")
    except FileNotFoundError:
        print("No models found. Train a new model.")


def save_model(trained_model, path="trained_model.pth"):
    torch.save(trained_model.state_dict(), path)
    print(f"Train Model saved in {path}")

def load_model(model, path="trained_model.pth"):
    try:
        model.load_state_dict(torch.load(path))
        print(f"Model loaded with {path}")
    except FileNotFoundError:
        print("No models found. Train a new model.")

import pygame

def play_notification_sound():
    pygame.mixer.init()
    pygame.mixer.music.load("notification_sound.mp3")  # substitua pelo caminho do seu arquivo de som
    pygame.mixer.music.play()

def show_desktop_notification(title, message):
  notification_title = f"[Task]: {title}"
  notification_message = f"Scheduled to {message}"
  notification.notify(
      title=notification_title,
      message=notification_message,
      timeout=10  # tempo em segundos que a notificação ficará visível
      )

def show_console_message(title, message):
    print(f"\nNotification: [Task]: {title} - [Message]: {message}")

def format_seconds(seconds):
    days, remainder = divmod(seconds, 86400)
    hours, remainder = divmod(remainder, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    return int(days), int(hours), int(minutes), int(seconds)

# **Implementation of Notifications Based on Due Dates:**
def schedule_notifications():
    # Check tasks with due date
    tasks_data = get_all_tasks()
    tasks_with_due_date = [(task[1], task[3]) for task in tasks_data if task[3] is not None]

    for task_description, due_date_str in tasks_with_due_date:
        # Convert due_date string to a datetime object
        due_date = datetime.strptime(due_date_str, '%Y-%m-%d %H:%M:%S')

        if due_date >= datetime.now():
            time_until_due = due_date - datetime.now()
            seconds_until_due = time_until_due.total_seconds()

            # Adjust as needed to notify just before the due date
            if seconds_until_due > 0:
                days, hours, minutes, seconds = format_seconds(seconds_until_due)
                notification_title = f"[Task]: {task_description}"
                notification_message = f"Scheduled to {due_date.strftime('%Y-%m-%d %H:%M')}"
                print(f"\nNotification: {notification_title} - [message]: {notification_message}\nFaltam: {days} dias, {hours} horas, {minutes} minutos, {seconds} segundos")
                if hours < 1 and minutes == 10:
                  message = print(f'Faltam: {minutes} minutos, and {seconds} segundos para Iniciar e/ou terminar ....')
                  # Chame esta função quando for o momento programado
                  #show_desktop_notification(task_description, notification_message)
                  # Exiba uma mensagem diretamente no console ou termina
                  show_console_message(notification_title, message)
                  # Chame esta função quando for o momento programado
                  play_notification_sound()

def get_user_due_date():
    while True:
        try:
            date_input = input("Enter the task due date (formato: YYYY-MM-DD): ")
            due_date = datetime.strptime(date_input, "%Y-%m-%d")
            return due_date
        except ValueError:
            print("Invalid date format. Try again.")


def add_due_date_to_task():
    # Model configuration
    input_size = 1
    hidden_size = 8
    output_size = 1
    model = NeuralNetwork(input_size, hidden_size, output_size)

    load_model(model)

    # After successfully training the model:
    save_model(model)
    description = input("Enter the task description: ")
    priority = get_user_priority()
    due_date = get_user_due_date()

    description_process = process_description(description)
    if priority is not None:
                priority_normalized = priority / 10  # Normalizing to [0, 1]

                # Insert task into database
                insert_task(description_process, priority_normalized, due_date)

                # Update the model with the new task
                tasks_data = get_all_tasks()
                print(f'Task Data: {tasks_data}')

                # Filter tasks with valid priorities
                valid_tasks = [(task[0], task[1], float(task[2])) for task in tasks_data if isinstance(task[2], (float, int))]
 
                # Training the model with existing data
                inputs = torch.tensor([[float(task[2])] for task in tasks_data], dtype=torch.float32)
                targets = torch.tensor([[task[2]] for task in tasks_data], dtype=torch.float32)

                if len(valid_tasks) >= 2:
                  filtered_tasks_data = [task for task in tasks_data if isinstance(task[2], (float, int))]
                  targets = torch.tensor([[task[2]] for task in filtered_tasks_data], dtype=torch.float32)
                  trained_model = train_neural_network(model, inputs, targets)
                  save_model(trained_model, path="trained_model.pth")
                  load_model(model, path="trained_model.pth")
                else:
                  print("At least 2 examples of tasks with valid priorities are required to train the model.")



# Main function
def main():
    # Database configuration
    create_database()

    # Configuração do modelo
    input_size = 1
    hidden_size = 8
    output_size = 1
    model = NeuralNetwork(input_size, hidden_size, output_size)

    load_model(model)

    # After successfully training the model:
    save_model(model)

    # Existing data in the database
    tasks_data = get_all_tasks()

    # Filter tasks with valid priorities
    valid_tasks = [(task[0], task[1], float(task[2])) for task in tasks_data if isinstance(task[2], (float, int))]

    if len(valid_tasks) >= 2:
        inputs = torch.tensor([[task[2]] for task in valid_tasks], dtype=torch.float32)
        # Check that priorities are numbers before creating targets
        try:
            targets = torch.tensor([[float(task[1])] for task in valid_tasks], dtype=torch.float32)
            trained_model = train_neural_network(model, inputs, targets)
        except (ValueError, TypeError):
            print("Error: Some priorities are not valid numbers.")
    else:
        print("At least 2 examples of tasks with valid priorities are required to train the model.")

    while True:
        display_menu()
        choice = get_user_choice()

        if choice == '1':
            description = input("Enter the task description: ")
            # Validate the task description
            if not description:
              print("Task description cannot be empty.")
              continue
          
            description_process = process_description(description)

            priority = get_user_priority()
            due_date = get_user_due_date()

            if priority is not None:
                priority_normalized = priority / 10 

                # Inserir tarefa no banco de dados
                insert_task(description_process, priority_normalized, due_date)

                # Update the model with the new task
                tasks_data = get_all_tasks()
                print(f'Task Data: {tasks_data}')

                # Filter tasks with valid priorities
                valid_tasks = [(task[0], task[1], float(task[2])) for task in tasks_data if isinstance(task[2], (float, int))]
 
                # Training the model with existing data
                inputs = torch.tensor([[float(task[2])] for task in tasks_data], dtype=torch.float32)
                targets = torch.tensor([[task[2]] for task in tasks_data], dtype=torch.float32) 

                if len(valid_tasks) >= 2:
                  filtered_tasks_data = [task for task in tasks_data if isinstance(task[2], (float, int))]
                  targets = torch.tensor([[task[2]] for task in filtered_tasks_data], dtype=torch.float32)
                  trained_model = train_neural_network(model, inputs, targets)
                  save_model(trained_model, path="trained_model.pth")
                  load_model(model, path="trained_model.pth")
            else:
                print("At least 2 examples of tasks with valid priorities are required to train the model.")
        elif choice == '2':
            # View tasks
            tasks_data = get_all_tasks()
            for task in tasks_data:
                print(f"ID: {task[0]} | Task: {task[1]} | Priority: {task[2]}")
        elif choice == '3':
          # Model training without adding new task
          tasks_data = get_all_tasks()

          # Filter tasks with valid priorities
          valid_tasks = [(task[0], task[1], float(task[2])) for task in tasks_data if isinstance(task[2], (float, int))]

          # Training the model with existing data
          inputs = torch.tensor([[float(task[2])] for task in tasks_data], dtype=torch.float32)
          targets = torch.tensor([[task[2]] for task in tasks_data], dtype=torch.float32)  

          if len(valid_tasks) >= 2:
            filtered_tasks_data = [task for task in tasks_data if isinstance(task[2], (float, int))]
            targets = torch.tensor([[task[2]] for task in filtered_tasks_data], dtype=torch.float32)
            epochs = int(input("Enter numbers Epochs: "))
            trained_model = train_neural_network(model, inputs, targets, epochs)
            save_model(trained_model, path="trained_model.pth")
            load_model(model, path="trained_model.pth")
          else:
            print("At least 2 examples of tasks with valid priorities are required to train the model.")

        elif choice == '4':
          add_due_date_to_task()
        elif choice == '5':
          schedule_notifications()
        elif choice == '6':
          #lines = clear_last_rows(0) # Uncomment line if you want the last (n) lines to be deleted when exiting
          print("Leaving! Thanks!! ... To the next!!!....")
          break
            
if __name__ == "__main__":
    main()