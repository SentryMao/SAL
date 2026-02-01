import numpy as np
import torch


class SPLLabelGenerator:
    """
    SPL (Segment Positional Labeling) generator with 8-class labels:
    - 0: false start positions
    - 1: false middle positions  
    - 2: false end positions
    - 3: single-frame false segments
    - 4: true start positions
    - 5: true middle positions
    - 6: true end positions
    - 7: single-frame true segments
    """
    
    @staticmethod
    def _seg2bd_label_new(seglabel):
        """
        Transfer binary label to 'T'/'F' and during lengths.
        Handle batch data with shape [batch_size, seq_length].
        """
        seglabel = seglabel.data.cpu().numpy().astype(int)
        
        # Handle batch data
        if len(seglabel.shape) == 2:
            batch_size, seq_length = seglabel.shape
            all_labels = []
            all_lengths = []
            
            for batch_idx in range(batch_size):
                seq_labels = []
                seq_lengths = []
                current_length = 1
                current_label = 'T' if seglabel[batch_idx, 0] == 1 else 'F'
                
                for i in range(1, seq_length):
                    if seglabel[batch_idx, i] == seglabel[batch_idx, i-1]:
                        current_length += 1
                    else:
                        seq_labels.append(current_label)
                        seq_lengths.append(current_length)
                        current_label = 'T' if seglabel[batch_idx, i] == 1 else 'F'
                        current_length = 1
                
                seq_labels.append(current_label)
                seq_lengths.append(current_length)
                all_labels.append(seq_labels)
                all_lengths.append(seq_lengths)
            
            return all_labels, all_lengths
        else:
            # Handle single sequence (original logic)
            labels = []
            lengths = []
            current_length = 1  
            current_label = 'T' if seglabel[0] == 1 else 'F'  
            for i in range(1, len(seglabel)):
                if seglabel[i] == seglabel[i-1]:  
                    current_length += 1 
                else:
                    labels.append(current_label)
                    lengths.append(current_length)
                    current_label = 'T' if seglabel[i] == 1 else 'F'
                    current_length = 1
            labels.append(current_label)
            lengths.append(current_length)
            return labels, lengths
    
    @staticmethod
    def seg2bd_label_new(labels, lengths):
        """
        Generate new boundary labels with 8 classes:
        - 0: false start positions
        - 1: false middle positions  
        - 2: false end positions
        - 3: single-frame false segments
        - 4: true start positions
        - 5: true middle positions
        - 6: true end positions
        - 7: single-frame true segments
        """
        # Handle batch data
        if isinstance(labels, list) and isinstance(labels[0], list):
            # Batch data: labels and lengths are lists of lists
            all_results = []
            for batch_idx in range(len(labels)):
                batch_labels = labels[batch_idx]
                batch_lengths = lengths[batch_idx]
                res = []
                
                for idx in range(len(batch_labels)):
                    label, length = batch_labels[idx], batch_lengths[idx]
                    temp = np.zeros(length)
                    
                    if label == 'F':  # False segments
                        if length == 1:
                            # Single frame false segment
                            temp[0] = 3
                        else:
                            # Multi-frame false segment
                            temp[0] = 0  # false start
                            temp[-1] = 2  # false end
                            # middle positions remain 0 (false middle)
                    else:  # True segments (label == 'T')
                        if length == 1:
                            # Single frame true segment
                            temp[0] = 7
                        else:
                            # Multi-frame true segment
                            temp[0] = 4  # true start
                            temp[-1] = 6  # true end
                            # middle positions remain 0 (will be set to 5)
                    
                    # Set middle positions for multi-frame segments
                    if length > 1:
                        if label == 'F':
                            temp[1:-1] = 1  # false middle
                        else:
                            temp[1:-1] = 5  # true middle
                    
                    res.extend(temp)
                
                all_results.append(np.array(res).reshape(-1, 1))
            return all_results
        else:
            # Single sequence (original logic)
            res = []
            for idx in range(len(labels)):
                label, length = labels[idx], lengths[idx]
                temp = np.zeros(length)
                
                if label == 'F':  # False segments
                    if length == 1:
                        # Single frame false segment
                        temp[0] = 3
                    else:
                        # Multi-frame false segment
                        temp[0] = 0  # false start
                        temp[-1] = 2  # false end
                        # middle positions remain 0 (false middle)
                else:  # True segments (label == 'T')
                    if length == 1:
                        # Single frame true segment
                        temp[0] = 7
                    else:
                        # Multi-frame true segment
                        temp[0] = 4  # true start
                        temp[-1] = 6  # true end
                        # middle positions remain 0 (will be set to 5)
                
                # Set middle positions for multi-frame segments
                if length > 1:
                    if label == 'F':
                        temp[1:-1] = 1  # false middle
                    else:
                        temp[1:-1] = 5  # true middle
                
                res.extend(temp)
            return np.array(res).reshape(-1, 1)


class TransitionLabelGenerator:
    """
    Transition label generator with 2-class labels:
    - 0: false segments
    - 1: true segments
    """
    @staticmethod
    def _seg2bd_label_new(seglabel):
        """
        Transfer binary label to 'T'/'F' and during lengths.
        Handle batch data with shape [batch_size, seq_length].
        """
        seglabel = seglabel.data.cpu().numpy().astype(int)
        
        # Handle batch data
        if len(seglabel.shape) == 2:
            batch_size, seq_length = seglabel.shape
            all_labels = []
            all_lengths = []
            
            for batch_idx in range(batch_size):
                seq_labels = []
                seq_lengths = []
                current_length = 1
                current_label = 'T' if seglabel[batch_idx, 0] == 1 else 'F'
                
                for i in range(1, seq_length):
                    if seglabel[batch_idx, i] == seglabel[batch_idx, i-1]:
                        current_length += 1
                    else:
                        seq_labels.append(current_label)
                        seq_lengths.append(current_length)
                        current_label = 'T' if seglabel[batch_idx, i] == 1 else 'F'
                        current_length = 1
                
                seq_labels.append(current_label)
                seq_lengths.append(current_length)
                all_labels.append(seq_labels)
                all_lengths.append(seq_lengths)
            
            return all_labels, all_lengths
        else:
            # Handle single sequence (original logic)
            labels = []
            lengths = []
            current_length = 1  
            current_label = 'T' if seglabel[0] == 1 else 'F'  
            for i in range(1, len(seglabel)):
                if seglabel[i] == seglabel[i-1]:  
                    current_length += 1 
                else:
                    labels.append(current_label)
                    lengths.append(current_length)
                    current_label = 'T' if seglabel[i] == 1 else 'F'
                    current_length = 1
            labels.append(current_label)
            lengths.append(current_length)
            return labels, lengths
    
    @staticmethod
    def seg2bd_label_new(labels, lengths):
        """
        Generate 2-class boundary labels (0/1) following `seg2bd_label` rules.

        Return:
        - For batched inputs: list of [T_i, 1] arrays
        - For single sequence: [T, 1] array
        """
        # Handle batch data
        if isinstance(labels, list) and isinstance(labels[0], list):
            # Batch data: labels and lengths are lists of lists
            all_results = []
            for batch_idx in range(len(labels)):
                batch_labels = labels[batch_idx]
                batch_lengths = lengths[batch_idx]
                res = []

                for idx in range(len(batch_labels)):
                    label, length = batch_labels[idx], batch_lengths[idx]
                    temp = np.zeros((length))
                    # mark boundaries by default
                    temp[0] = 1
                    temp[-1] = 1
                    # exceptions for 'T' segments at sequence edges
                    if idx == 0 and label == 'T':
                        temp[0] = 0
                    if idx == len(batch_labels) - 1 and label == 'T' and length > 1:
                        temp[-1] = 0
                    res.extend(temp)

                all_results.append(np.array(res).reshape(-1, 1))
            return all_results
        else:
            # Single sequence
            res = []
            for idx in range(len(labels)):
                label, length = labels[idx], lengths[idx]
                temp = np.zeros((length))
                temp[0] = 1
                temp[-1] = 1
                if idx == 0 and label == 'T':
                    temp[0] = 0
                if idx == len(labels) - 1 and label == 'T' and length > 1:
                    temp[-1] = 0
                res.extend(temp)
            return np.array(res).reshape(-1, 1)

