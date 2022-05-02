import random
import pandas as pd
import numpy as np


class coinworld:

    def __init__(self):
        columns = ['close', 'volume']

        self.btc = pd.read_csv('raw_data/btcusd.csv',usecols=columns)
        self.btc.dropna(inplace=True)
        self.btc_price = self.btc.values[:,:,np.newaxis]

        self.eos = pd.read_csv('raw_data/eosusd.csv',usecols=columns)
        self.eos.dropna(inplace=True)
        self.eos_price = self.eos.values[:,:,np.newaxis]

        self.eth = pd.read_csv('raw_data/ethusd.csv',usecols=columns)
        self.eth.dropna(inplace=True)
        self.eth_price = self.eth.values[:,:,np.newaxis]

        self.ltc = pd.read_csv('raw_data/ltcusd.csv',usecols=columns)
        self.ltc.dropna(inplace=True)
        self.ltc_price = self.ltc.values[:,:,np.newaxis]

        self.xrp = pd.read_csv('raw_data/xrpusd.csv',usecols=columns)
        self.xrp.dropna(inplace=True)
        self.xrp_price = self.xrp.values[:,:,np.newaxis]

        self.ACTION_SPACE_SIZE = 5

        self.btc = self.normalize(self.btc)
        self.eos = self.normalize(self.eos)
        self.eth = self.normalize(self.eth)
        self.ltc = self.normalize(self.ltc)
        self.xrp = self.normalize(self.xrp)

        self.btc = self.btc.values[:,:,np.newaxis]
        self.eos = self.eos.values[:,:,np.newaxis]
        self.eth = self.eth.values[:,:,np.newaxis]
        self.ltc = self.ltc.values[:,:,np.newaxis]
        self.xrp = self.xrp.values[:,:,np.newaxis]
        return

    def normalize(self,df):
        result = df.copy()
        for feature_name in df.columns:
            max_value = df[feature_name].max()
            min_value = df[feature_name].min()
            result[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)
        return result

    def random_sampler(self,coin,coin_price):
        coin_size = coin.shape[0]
        sampler  = random.randint(0,(coin_size-1441))
        random_samples = coin[sampler:(sampler+1441)]
        random_price_samples = coin_price[sampler:(sampler+1441)]

        return random_samples, random_price_samples

    def reset(self):
        ## Arange the world state

        self.debug = 'Environment Resetted' + '\n'

        self.sample_btc , self.sample_btc_price = self.random_sampler(self.btc,
                                                                 self.btc_price)
        self.sample_eos , self.sample_eos_price = self.random_sampler(self.eos,
                                                                 self.eos_price)
        self.sample_eth , self.sample_eth_price = self.random_sampler(self.eth,
                                                                 self.eth_price)
        self.sample_ltc , self.sample_ltc_price = self.random_sampler(self.ltc,
                                                                 self.ltc_price)
        self.sample_xrp , self.sample_xrp_price = self.random_sampler(self.xrp,
                                                                 self.xrp_price)


        self.debug += (('sample_btc shape:' + str(self.sample_btc.shape) +
                        '\n' + 'sample_btc_price shape:' + str(self.sample_btc_price.shape) + '\n'))
        #print(self.sample_btc.shape, self.sample_btc_price[0:5])
        #print(self.sample_btc_price.shape, self.sample_btc_price.shape[0:5])

        self.oracle_vision = np.concatenate((self.sample_btc, self.sample_eos,
                                             self.sample_eth, self.sample_ltc,
                                             self.sample_xrp), axis=2)

        self.debug += ('oracle_vision shape:' + str(self.oracle_vision.shape) + '\n')

        self.oracle_vision_prices = np.concatenate((self.sample_btc_price,
                                                    self.sample_eos_price,
                                                    self.sample_eth_price,
                                                    self.sample_ltc_price,
                                                    self.sample_xrp_price),
                                                    axis=2)
        self.debug += ('oracle_vision_prices shape:' + str(self.oracle_vision_prices.shape) + '\n')

        self.observation = self.oracle_vision[:60]
        self.debug += ('observation shape:' + str(self.observation.shape) + '\n')

        self.prices = self.oracle_vision_prices[59:,0]
        self.debug += ('prices shape:' + str(self.prices.shape) + '\n')

        self.done = 0
        self.hold_count = 0
        ## Arange the inventory
        self.oldnetworth = 1000
        self.credit = 1000
        self.coin_info = np.zeros(shape=(5))
        self.btc_owned = 0
        self.eos_owned = 0
        self.eth_owned = 0
        self.ltc_owned = 0
        self.xrp_owned = 0
        self.counter = 0
        self.coin_info = np.zeros(shape=(5))
        self.hold_reward = 0
        self.have_coins = 0

        return [self.observation[:,0,:], self.observation[:,1,:],self.coin_info]

    def calculate_networth(self,future,buy):
        self.debug += ('calculate networth future:' + str(future) + 'buy:' + str(buy) + '\n')
        networth = 0
        networth += self.credit
        self.debug += ('networth add credit:' + str(self.credit) + '\n')
        if future:
            pt = 15
        else:
            pt = 0

        networth += self.btc_owned * self.prices[(15 * self.counter) + pt,0]
        self.debug += ('networth add btc, btc owned:' + str(self.btc_owned) + 'btc_price:'+ str(self.prices[(15 * self.counter) + pt,0]) + 'total:' + str(self.btc_owned * self.prices[(15 * self.counter) + pt,0]) + '\n')
        networth += self.eos_owned * self.prices[(15 * self.counter) + pt,1]
        self.debug += ('networth add eos, eos owned:' + str(self.eos_owned) + 'eos_price:'+ str(self.prices[(15 * self.counter) + pt,1]) + 'total:' + str(self.eos_owned * self.prices[(15 * self.counter) + pt,1]) + '\n')
        networth += self.eth_owned * self.prices[(15 * self.counter) + pt,2]
        self.debug += ('networth add eth, eth owned:' + str(self.eth_owned) + 'eth_price:'+ str(self.prices[(15 * self.counter) + pt,2]) + 'total:' + str(self.eth_owned * self.prices[(15 * self.counter) + pt,2]) + '\n')
        networth += self.ltc_owned * self.prices[(15 * self.counter) + pt,3]
        self.debug += ('networth add ltc, ltc owned:' + str(self.ltc_owned) + 'ltc_price:'+ str(self.prices[(15 * self.counter) + pt,3]) + 'total:' + str(self.ltc_owned * self.prices[(15 * self.counter) + pt,3]) + '\n')
        networth += self.xrp_owned * self.prices[(15 * self.counter) + pt,4]
        self.debug += ('networth add xrp, xrp owned:' + str(self.xrp_owned) + 'xrp_price:'+ str(self.prices[(15 * self.counter) + pt,4]) + 'total:' + str(self.xrp_owned * self.prices[(15 * self.counter) + pt,4]) + '\n')

        if buy == 1:
            networth = networth * (0.9975 ** 2)
            self.debug += ('networth add comission:' + str(networth) + '\n')

        return networth

    def trade_coin(self,action):
        self.debug += ('Calculating networth for trade credit' + '\n')
        self.credit = self.calculate_networth(future=0,buy=1)
        self.debug += ('back to trade coin, credit:' + str(self.credit) + '\n')
        self.coin_info = np.zeros(shape=(5))
        self.btc_owned = 0
        self.eos_owned = 0
        self.eth_owned = 0
        self.ltc_owned = 0
        self.xrp_owned = 0

        if action == 0:
             self.btc_owned = self.credit / self.prices[15 * self.counter,0]
             self.debug += ('action:' + str(action) + 'btc purchased at price:' + str(self.prices[15 * self.counter,0]) + 'total btc:' + str(self.btc_owned) + '\n')
             #self.purchase_price = self.prices[15 * self.counter,0]
             self.coin_info[0] = 1

        elif action == 1:
             self.eos_owned = self.credit / self.prices[15 * self.counter,1]
             self.debug += ('action:' + str(action) + 'eos purchased at price:' + str(self.prices[15 * self.counter,1]) + 'total eos:' + str(self.eos_owned) + '\n')
             #self.purchase_price = self.prices[15 * self.counter,1]
             self.coin_info[1] = 1
        elif action == 2:
             self.eth_owned = self.credit / self.prices[15 * self.counter,2]
             self.debug += ('action:' + str(action) + 'eth purchased at price:' + str(self.prices[15 * self.counter,2]) + 'total eth:' + str(self.eth_owned) + '\n')
             #self.purchase_price = self.prices[15 * self.counter,2]
             self.coin_info[2] = 1
        elif action == 3:
             self.ltc_owned = self.credit / self.prices[15 * self.counter,3]
             self.debug += ('action:' + str(action) + 'ltc purchased at price:' + str(self.prices[15 * self.counter,3]) + 'total ltc:' + str(self.ltc_owned) + '\n')
             #self.purchase_price = self.prices[15 * self.counter,3]
             self.coin_info[3] = 1
        elif action == 4:
             self.xrp_owned = self.credit / self.prices[15 * self.counter,4]
             self.debug += ('action:' + str(action) + 'xrp purchased at price:' + str(self.prices[15 * self.counter,4]) + 'total xrp:' + str(self.xrp_owned) + '\n')
             #self.purchase_price = self.prices[15 * self.counter,4]
             self.coin_info[4] = 1

        self.credit = 0
        self.have_coins = 1
        return


    def step(self,action):
        self.debug += ('Step' + str(self.counter) + '\n')
        # Check if the action is to hold the coin
        if action == np.argmax(self.coin_info) and self.have_coins == 1:
            self.traded = 0
            self.debug += ('No trade' + '\n')

        else:
            self.debug += ('Trade' + '\n')
       	    self.trade_coin(action)
            self.traded = 1
            self.debug += ('Trade done' + '\n')

        self.debug += ('Calculating newnetworth' + '\n')
        self.newnetworth = self.calculate_networth(future=1, buy=0)
        self.difference = self.newnetworth - self.oldnetworth
        self.debug += ('Calculating difference, futurenetworth:' + str(self.newnetworth) + 'oldnetworth:' + str(self.oldnetworth) + 'difference:' + str(self.difference) + '\n')

        if self.difference < 0 and self.traded == 0:
            self.reward = -2
            self.debug += ('reward: -2' + '\n')

        elif self.difference < 0 and self.traded == 1:
            self.reward = -1
            self.debug += ('reward: -1' + '\n')

        elif self.difference > 0 and self.traded == 0:
            self.reward = 1
            self.debug += ('reward: 1' + '\n')

        elif self.difference > 0 and self.traded == 1:
            self.reward = 2
            self.debug += ('reward: 2' + '\n')

        self.oldnetworth = self.newnetworth
        self.counter +=1
        self.observation = self.oracle_vision[(15*self.counter):
                                              ((15*self.counter)+60)]
        self.debug += ('Updating observation, observation shape:' + str(self.observation.shape))
        if self.counter == 92:
            self.done = 1

        return [self.observation[:,0,:], self.observation[:,1,:],
                self.coin_info], self.reward, self.difference, self.done