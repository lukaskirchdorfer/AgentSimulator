import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Concatenate
from tensorflow.keras.optimizers import Adam
import copy


# Parameters
n = 5  # length of the sequence (n-grams)
activity_col = 'activity_name'
resource_col = 'agent'
case_id_col = 'case_id'

def prepare_data_lstm(df, n=5):
    """
    Prepares data for LSTM training by encoding activities and resources, and creating n-grams.

    Parameters:
    - df: pd.DataFrame, the dataframe containing the data.
    - n: int, the length of the sequence (n-grams).
    
    Returns:
    - X_activities: np.array of shape (num_samples, sequence_length), encoded activity sequences.
    - X_resources: np.array of shape (num_samples, sequence_length), encoded resource sequences.
    - y_activity: np.array of shape (num_samples,), target next activity for each sequence.
    - y_resource: np.array of shape (num_samples,), target next resource for each sequence.
    """
    # Sort the dataframe by case_id and start_timestamp to maintain the order of activities in cases
    df = copy.deepcopy(df)
    df = df.sort_values(by=[case_id_col, 'start_timestamp'])
    
    # Initialize label encoders
    activity_encoder = LabelEncoder()
    resource_encoder = LabelEncoder()

    # Fit the encoders on the activity and resource columns
    df['activity_encoded'] = activity_encoder.fit_transform(df[activity_col])
    df['resource_encoded'] = resource_encoder.fit_transform(df[resource_col])
    
    # Group data by case_id
    grouped_cases = df.groupby(case_id_col)
    
    X_activities = []
    X_resources = []
    y_activity = []
    y_resource = []

    # Loop through each case to create padded sequences (n-grams)
    for case_id, group in grouped_cases:
        activities = group['activity_encoded'].values
        resources = group['resource_encoded'].values
        
        # Stop one step before the last activity to avoid forming a sequence with 'None' target
        for i in range(len(activities) - 1): 
            # Define current activity/resource sequence with padding if needed
            current_activities = activities[max(0, i-n+1):i+1]
            current_resources = resources[max(0, i-n+1):i+1]
            
            # Pad sequences from the left if they are shorter than n
            current_activities_padded = pad_sequences([current_activities], maxlen=n, padding='pre', value=-1)[0]
            current_resources_padded = pad_sequences([current_resources], maxlen=n, padding='pre', value=-1)[0]
            
            X_activities.append(current_activities_padded)
            X_resources.append(current_resources_padded)
            
            # The target is the next activity
            y_activity.append(activities[i+1])
            y_resource.append(resources[i+1])
    # Convert lists to numpy arrays
    X_activities = np.array(X_activities)
    X_resources = np.array(X_resources)
    y_activity = np.array(y_activity)
    y_resource = np.array(y_resource)
    
    return X_activities, X_resources, y_activity, y_resource, activity_encoder, resource_encoder


def train_lstm_model(X_activities, X_resources, y_activity, y_resource, activity_vocab_size, resource_vocab_size, 
                     agent_centric=True, embedding_dim=32, lstm_units=64, n_epochs=20, batch_size=32):
    """
    Trains an LSTM model to predict the next activity given sequences of (activity, resource) pairs.
    
    Parameters:
    - X_activities: np.array of shape (num_samples, sequence_length), encoded activity sequences.
    - X_resources: np.array of shape (num_samples, sequence_length), encoded resource sequences.
    - y_activity: np.array of shape (num_samples,), target next activity for each sequence.
    - y_resource: np.array of shape (num_samples,), target next resource for each sequence.
    - activity_vocab_size: int, the number of unique activities (for the embedding layer).
    - resource_vocab_size: int, the number of unique resources (for the embedding layer).
    - embedding_dim: int, dimension of the embedding space.
    - lstm_units: int, the number of LSTM units in the LSTM layer.
    - n_epochs: int, the number of epochs to train the model.
    - batch_size: int, the batch size for training.
    
    Returns:
    - model: Trained LSTM model.
    """
    
    if agent_centric == True:
        # Input layers for activities and resources
        activity_input = Input(shape=(X_activities.shape[1],), name='activity_input')
        resource_input = Input(shape=(X_resources.shape[1],), name='resource_input')
        # Embedding layers for activities and resources
        activity_embedding = Embedding(input_dim=activity_vocab_size, output_dim=embedding_dim, name='activity_embedding')(activity_input)
        resource_embedding = Embedding(input_dim=resource_vocab_size, output_dim=embedding_dim, name='resource_embedding')(resource_input)
        # Concatenate the embeddings
        concat = Concatenate(name='concatenated')([activity_embedding, resource_embedding])

        inputs = [activity_input, resource_input]
        x = {'activity_input': X_activities, 'resource_input': X_resources}

        # LSTM layer
        lstm_output = LSTM(units=lstm_units, name='lstm_layer')(concat)

        # Dense output layer with softmax activation to predict the next activity
        output_activity = Dense(activity_vocab_size, activation='softmax', name='output_activity')(lstm_output)
        output_resource = Dense(resource_vocab_size, activation='softmax', name='output_resource')(lstm_output)
        output = [output_activity, output_resource]

        loss = {'output_activity': 'sparse_categorical_crossentropy', 'output_resource': 'sparse_categorical_crossentropy'}
        metrics = ['accuracy', 'accuracy']

        y = {'output_activity': y_activity, 'output_resource': y_resource}

    else:
        activity_input = Input(shape=(X_activities.shape[1],), name='activity_input')
        activity_embedding = Embedding(input_dim=activity_vocab_size, output_dim=embedding_dim, name='activity_embedding')(activity_input)
        concat = activity_embedding

        inputs = [activity_input]
        x = {'activity_input': X_activities}
    
        # LSTM layer
        lstm_output = LSTM(units=lstm_units, name='lstm_layer')(concat)
        
        # Dense output layer with softmax activation to predict the next activity
        output = Dense(activity_vocab_size, activation='softmax', name='output_activity')(lstm_output)

        loss = {'output_activity': 'sparse_categorical_crossentropy'}
        metrics = ['accuracy']

        y = {'output_activity': y_activity}
    
    # Create the model
    model = Model(inputs=inputs, outputs=output)
    
    # Compile the model
    model.compile(optimizer=Adam(), loss=loss, metrics=metrics)
    
    # Train the model
    model.fit(x, y, epochs=n_epochs, batch_size=batch_size, validation_split=0.2)
    
    return model


def predict_next_step(model, activities, resources, activity_encoder, resource_encoder, n=5):
    """
    Predicts the next activity given a sequence of activities and resources.
    
    Parameters:
    - model: Trained LSTM model.
    - activities: List of past activities (in their original categorical form).
    - resources: List of past resources (agents) corresponding to the activities.
    - activity_encoder: LabelEncoder for the activities (used during training).
    - resource_encoder: LabelEncoder for the resources (used during training).
    - n: Length of the expected sequence (n-gram window size).
    
    Returns:
    - next_activity: The predicted next activity in its original categorical form.
    - next_resource_distribution: The predicted likelihood for each resource in its original categorical form.
    """

    if resource_encoder != None:
        # Encode the activities and resources using the encoders
        encoded_activities = activity_encoder.transform(activities)
        encoded_resources = resource_encoder.transform(resources)
        
        # Ensure the sequences are the correct length (pad if necessary)
        padded_activities = pad_sequences([encoded_activities], maxlen=n, padding='pre', value=-1)[0]
        padded_resources = pad_sequences([encoded_resources], maxlen=n, padding='pre', value=-1)[0]

        padded_activities = np.array(padded_activities)
        padded_resources = np.array(padded_resources)

        # prediction = model.predict([padded_activities[np.newaxis, :], padded_resources[np.newaxis, :]])[0]
        # Get both activity and resource predictions
        activity_pred, resource_pred = model.predict([padded_activities[np.newaxis, :], padded_resources[np.newaxis, :]])
        
        # Sample from both probability distributions
        predicted_activity_index = np.random.choice(len(activity_pred[0]), p=activity_pred[0])
        # predicted_resource_index = np.random.choice(len(resource_pred[0]), p=resource_pred[0])
        # get the predicted likelihood for each resource, decode it back to original form
        next_resource_distribution = {resource_encoder.inverse_transform([i])[0]: resource_pred[0][i] for i in range(len(resource_pred[0]))}
        next_resource_distribution = sorted(next_resource_distribution.keys(), key=lambda x: next_resource_distribution[x], reverse=True)
        
        # Decode both predictions
        next_activity = activity_encoder.inverse_transform([predicted_activity_index])[0]
        # next_resource = resource_encoder.inverse_transform([predicted_resource_index])[0]
        
        return next_activity, next_resource_distribution
        
    else:
        encoded_activities = activity_encoder.transform(activities)
        padded_activities = pad_sequences([encoded_activities], maxlen=n, padding='pre', value=-1)[0]
        padded_activities = np.array(padded_activities)
        prediction = model.predict([padded_activities[np.newaxis, :]])[0]

        predicted_activity_index = np.random.choice(len(prediction), p=prediction)

        # Decode the predicted activity back to its original form
        next_activity = activity_encoder.inverse_transform([predicted_activity_index])[0]
        
        return next_activity, None
