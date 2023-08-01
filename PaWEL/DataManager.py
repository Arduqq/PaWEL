import csv


class DataManager:
    """
    Simple class to store the retrieved data

    Attributes:
        output:      Result data in form of a table
        heart_data:  Data from the FFT
    """
    output = None
    heart_data = None

    def __init__(self, output, heart_data):
        self.output = output
        self.heart_data = heart_data

    def save_to_csv(self):
        """Saves all of the data to a csv file for further analysis"""
        with open(self.output, 'w', newline='') as csvfile:
            fieldnames = ['time', 'heart_rate', 'pupil_size']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for peak in self.heart_data:
                writer.writerow({'time': peak[0], 'heart_rate': peak[1], 'pupil_size': 5})
                csvfile.flush()
