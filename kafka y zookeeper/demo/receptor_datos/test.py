
from mplfinance.original_flavor import candlestick_ohlc
class CandleStickPlotter:
    import matplotlib.ticker as mticker
    import matplotlib.pyplot as plt
    
    import pandas as pd
    import matplotlib.dates as mpdates

    def __init__(self):
        self.plt.style.use('dark_background')

        # creating Subplots
        self.fig, self.ax = self.plt.subplots()

        # allow grid
        self.ax.grid(True)

        # Setting labels 
        self.ax.set_xlabel('Date')
        self.ax.set_ylabel('Price')

        # setting title
        self.plt.title('Prices For the Period 01-07-2020 to 15-07-2020')

        #self.__inicializar_grafica()
    
    def __inicializar_grafica(self, save=True):
        # show the plot
        if save:
            self.plt.savefig("hist.jpg")
        else:
            self.plt.show()
            self.plt.draw()

    def __prepare_data_for_plot(self, df):
        # extracting Data for plotting
        df = df[['Date', 'Open', 'High', 
                'Low', 'Close']]
        
        # convert into datetime object
        df['Date'] = self.pd.to_datetime(df['Date'])
        
        # apply map function
        df['Date'] = df['Date'].map(self.mpdates.date2num)
        return df
    
    def update_graph(self, df):

        df = self.__prepare_data_for_plot(df)

        # plotting the data
        candlestick_ohlc(self.ax, df.values, width = 0.6,
                    colorup = 'green', colordown = 'red', 
                    alpha = 0.8)

        # Formatting Date
        date_format = self.mpdates.DateFormatter('%d-%m-%Y')
        self.ax.xaxis.set_major_formatter(date_format)
        self.fig.autofmt_xdate()
        
        self.fig.tight_layout()

        graph.__inicializar_grafica()

    def cerrar_grafica(self):
        self.plt.close()

class Pintar:
    import matplotlib.pyplot as plt
    from PIL import Image
    def __init__(self):
        self.plt.style.use('dark_background')
        self.fig, self.ax = self.plt.subplots()
        self.ax.grid(False)


        image = self.Image.open('hist.jpg')
        self.plt.imshow(image)  
        self.plt.show()

if __name__ == '__main__':
    

    import time
    
    import pandas as pd
    for i in range(10,20,1):
        df = pd.read_csv('data.csv')  # https://www1.nseindia.com/products/content/equities/indices/historical_index_data.htm
        df = df[:i]
        print(df.to_json())

        graph = CandleStickPlotter()
        graph.update_graph(df)
        time.sleep(5)

    
    
    
    
    