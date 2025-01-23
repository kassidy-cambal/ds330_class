import pandas as pd
from zipfile import ZipFile # allows us to pass open file to csv

# buffer = open file 

# reading in wine quality 
def read_wq(path: str) -> pd.DataFrame:
    """Read in the wine quality dataset from zip file

    Args:
        path (str): location of file on computer

    Returns:
        pd.DataFrame: complete wine quality dataset
    """
    zf = ZipFile(path)
    df = pd.read_csv(zf.open('winequality-white.csv'), sep = ';')
    return df

# anything after if __name__ == "__main__" will run if this file is ran
# good to use this if you have test code 
if __name__ == '__main__':
    df = read_wq('data/wine_quality.zip')

    print(1)

