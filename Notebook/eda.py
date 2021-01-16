import os
import warnings
warnings.filterwarnings("ignore")
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns



class data_process:

    def __init__(self, data):
        self.data = data
        self.count = data.nunique()
        self.va_pct = round(self.count / (data.shape[0]) * 100, 4)
        self.nan_p = nan_p = round(data.isnull().sum() / data.shape[0] * 100, 4)

    def data_info(self):
        frame = pd.DataFrame(zip(self.count, self.va_pct, self.nan_p),
                             index=self.count.index,
                             columns=['counts', 'unique_value_pct', 'nan_pct']).reset_index().rename(
            columns={'index': 'column'})

        frame['data_type'] = self.data.dtypes.tolist()
        return frame.sort_values('counts', ascending=False)

    def convert_datetime(self, datecol):
        self.data[datecol] = pd.to_datetime(self.data[datecol], format='%Y-%m-%d')

    #     def int_to_str(self, col):
    #         self.data[col] = self.data[col].astype('str')
    
    def str_to_int(self, col):
        # a = public holiday, b = Easter holiday, c = Christmas, 0 = None
        self.data[col] = self.data[col].map(lambda x: 1 if x=='a' else 2 if x=='b' else 3 if x=='c' else 0)

    def convert_dtype(self):
        catcolumns = self.data.select_dtypes(include=['object']).columns.tolist()
        return pd.get_dummies(self.data, columns=catcolumns)
    
    

    def pivot_table(self, index, aggfunc):
        return pd.pivot_table(self.data, index=index, aggfunc=aggfunc)

    
    
    

class Sales_EDA:
    def __init__(self, data):
        self.data = data

    def nlargest_ID(self, N):
        topN_storeID = self.data.groupby('Store')['Sales'].sum().nlargest(N).index.tolist()
        return topN_storeID

    def nsmallest_ID(self, N):
        bottomN_storeID = self.data.groupby('Store')['Sales'].sum().nsmallest(N).index.tolist()
        return bottomN_storeID

    def topN_weekly_salesplot(self, N):
        plt.figure(figsize=(12, 7))
        for storeid in self.nlargest_ID(N):
            self.data.loc[self.data.Store == storeid].set_index('Date').Sales.resample('w').sum().plot(
                label=f'Store {storeid}')
        plt.legend()
        plt.title(f'Weekly Sales of Top {N} Store', fontsize=15)
        plt.xlabel('Date', fontsize=15)
        plt.ylabel('Weekly Sales', fontsize=15)
        plt.show()

    def bottomN_weekly_salesplot(self, N):
        plt.figure(figsize=(12, 7))
        for storeid in self.nsmallest_ID(N):
            self.data.loc[self.data.Store == storeid].set_index('Date').Sales.resample('w').sum().plot(
                label=f'Store {storeid}')
        plt.legend()
        plt.title(f'Weekly Sales of Bottom {N} Store', fontsize=15)
        plt.xlabel('Date', fontsize=15)
        plt.ylabel('Weekly Sales', fontsize=15)
        plt.show()

    def topN_daily_violinplot(self, N):
        plt.figure(figsize=(12, 7))
        N_data = self.data.loc[self.data.Store.isin(self.nlargest_ID(N))]
        sns.violinplot(x='Sales', y='Store',
                       data=N_data, order=self.nlargest_ID(N),
                       orient='h')
        plt.title('Daily Sales of Top10 Stores', fontsize=15)
        plt.xlabel('Sales', fontsize=15)
        plt.ylabel('Store', fontsize=15)
        plt.show()

    def bottomN_daily_violinplot(self, N):
        plt.figure(figsize=(12, 7))
        b_N_data = self.data.loc[self.data.Store.isin(self.nsmallest_ID(N))]
        sns.violinplot(x='Sales', y='Store',
                       data=b_N_data, order=self.nsmallest_ID(N),
                       orient='h')

        # sns.violinplot(x='Store', y='Sales', data=df.loc[df.Store.isin(self.nsmallest_ID(N))])
        plt.title('Daily Sales of Bottom Stores', fontsize=15)
        plt.xlabel('Daily Sales', fontsize=15)
        plt.ylabel('Store ID', fontsize=15)
        plt.show()


        
        
class sumby:
    def __init__(self, data, groupbycol, aggcol):
        self.data = data
        self.groupbycol = groupbycol
        self.aggcol = aggcol

    def get_sum_pct(self):
        total = self.data.groupby(self.groupbycol)[self.aggcol].sum()
        total = pd.DataFrame(total)
        total['pct'] = (total[self.aggcol] / total[self.aggcol].sum()) * 100
        total = total.sort_values(self.aggcol, ascending=False)
        total = total.reset_index()
        return total

    def histgram(self, col):
        self.get_sum_pct()[col].hist()
        per50 = self.get_sum_pct()[col].describe()['50%']
        mean = self.get_sum_pct()[col].describe()['mean']
        plt.axvline(per50, color='r')
        plt.axvline(mean, color='r', linestyle='-.')
        plt.legend(['median', 'mean'])
        plt.title(f'Total {self.aggcol} by {self.groupbycol}', fontsize=15)
        plt.show()

    def topN_barplot(self, N):
        bardata = self.get_sum_pct().set_index(self.groupbycol)
        plt.figure(figsize=(12, 7))
        sns.barplot(y=bardata.Sales.nlargest(N).index,
                    x=bardata.Sales.nlargest(N).values,
                    order=bardata.Sales.nlargest(N).index,
                    orient='h')
        plt.title(f'Total {self.aggcol} of Top {N} Stores', fontsize=15)
        plt.xlabel(self.aggcol, fontsize=15)
        plt.ylabel(self.groupbycol, fontsize=15)

    def topN_sum_share(self, N):
        top10sum = round(self.get_sum_pct().iloc[:N].sum()[1], 0)
        top10pct = round(self.get_sum_pct().iloc[:N].sum()[2], 3)
        if len(self.data) > 1115:
            print(f'Top {N} {self.groupbycol} Total {self.aggcol}: {top10sum}')
            print(f'Top {N} {self.groupbycol} Total {self.aggcol} Share in All Stores: {top10pct}%')
        else:
            storetype = list(set(self.data.StoreType))[0].upper()
            print(f'{storetype} Type Store Top {N} {self.groupbycol} Total {self.aggcol}: {top10sum}')
            print(f'{storetype} Type Store Top {N} {self.groupbycol} Total {self.aggcol} Share: {top10pct}%')


            
            
class individual_store:
    def __init__(self, data, indexcol, storeid):
        self.data = data
        self.indexcol = indexcol
        self.storeid = storeid
        self.total_sales = sumby(self.data, 'Store', 'Sales').get_sum_pct()
        self.id_total = self.total_sales.loc[self.total_sales.Store == self.storeid]
        self.storesales = self.id_total.iloc[0][1]
        self.ranking = self.id_total.index[0] + 1
        self.share = round(self.id_total.iloc[0][2], 3)
        self.opendays = self.data.loc[self.data.Store == self.storeid].set_index(self.indexcol).Open.value_counts()[1]
        if len(self.data.loc[self.data.Store == self.storeid].set_index(self.indexcol).Open.value_counts().index) == 2:
            self.closeddays = \
            self.data.loc[self.data.Store == self.storeid].set_index(self.indexcol).Open.value_counts()[0]
        else:
            self.closeddays = 0

    def store_opendays(self):
        print(f'Store {self.storeid} open days: {self.opendays}')
        print(f'Store {self.storeid} closed days: {self.closeddays}')

    def get_sales_share_ranking(self):
        storenumber = len(set(self.data.Store))
        print(f'Total Sales of Store {self.storeid} is {self.storesales}')
        print(f'Sales share among {storenumber} stores : {self.share}%')
        print(f'Ranking among {storenumber} stores : {self.ranking}')

    def lineplot(self, col):
        plt.figure(figsize=(12, 7))
        dailydata = self.data.loc[self.data.Store == self.storeid].set_index('Date')[col]
        plt.plot(dailydata)
        plt.plot(dailydata.index, dailydata.rolling(14).mean(), 'red')
        plt.plot(dailydata.resample('w').sum(), 'green')
        plt.legend([f'Daily {col}', 'Rolling mean (14D)', f'Weekly {col}'])
        plt.title(f'{col} of store {self.storeid}', fontsize=15)
        plt.show()

    def store_report(self, comdata):
        storetype = comdata.loc[comdata.Store == self.storeid].StoreType.iloc[0]
        assortmenttype = comdata.loc[comdata.Store == self.storeid].Assortment.iloc[0]
        storetypedata = sumby(comdata.loc[comdata.StoreType == storetype], 'Store', 'Sales').get_sum_pct()
        assortttypedata = sumby(comdata.loc[comdata.Assortment == assortmenttype], 'Store',
                                'Sales').get_sum_pct()
        ranking2 = storetypedata.loc[storetypedata.Store == self.storeid].index[0] + 1
        ranking3 = assortttypedata.loc[assortttypedata.Store == self.storeid].index[0] + 1
        storenumber2 = len(storetypedata)
        storenumber3 = len(assortttypedata)
        share2 = round(storetypedata.iloc[0][2], 3)
        share3 = round(assortttypedata.iloc[0][2], 3)

        return pd.DataFrame([self.storeid, storetype.upper(), assortmenttype.upper(), self.opendays,
                             self.closeddays, self.storesales, self.share, share2, share3,
                             self.ranking, ranking2, ranking3],
                            index=['Storeid', 'Store Type', 'Assortment', 'Open Days',
                                   'Closed Days', 'Total Sales', 'Share in All Store (%)',
                                   f'Share in {storetype.upper()} Type (%)',
                                   f'Share in {assortmenttype.upper()} Assortment (%)',
                                   'Ranking in All Store',
                                   f'Ranking in {storetype.upper()} Type',
                                   f'Ranking in {assortmenttype.upper()} Assortment'],
                            columns=['Detail'])

    
    

    
    
class combined_storedata:
    def __init__(self, data1, data2):
        self.data1 = data1
        self.data2 = pd.read_csv(data2)

    def data2_info(self):
        return self.data2

    def process_data1(self):
        # change the index of data1 from storeid to number
        self.data1 = self.data1.reset_index()
        # sort store id in ascending order
        self.data1 = self.data1.sort_values('Store').reset_index(drop=True)
        return self.data1

    def process_data2(self):
        # drop duplcated column Store
        self.data2 = self.data2.drop('Store', axis=1)
        return self.data2

    def concatdf(self):
        # combine two data set 
        storedata = pd.concat([self.process_data1(), self.process_data2()], axis=1)
        return storedata


    
    
    
    
class combined_store_EDA:
    def __init__(self, data):
        self.data = data

    def storetype_count(self):
        print(self.data.StoreType.value_counts())

    def storetype_share_pie(self):
        plt.figure(figsize=(12, 7))
        self.data.groupby('StoreType')['Sales'].sum().plot(kind='pie', autopct='%.2f%%')
        plt.title('Sales Share by Store Type', fontsize=15)
        plt.show()

    def average_sales_by_storetype(self):
        average_sales_by_storetype = self.data.groupby('StoreType')[
                                         'Sales'].sum() / self.data.StoreType.value_counts().sort_index()
        sns.barplot(x=average_sales_by_storetype.values, y=average_sales_by_storetype.index, orient='h')
        plt.title('Sales Per Store by Store Type', fontsize=15)
        plt.xlabel('Sales', fontsize=15)
        plt.ylabel('StoreType', fontsize=15)
        plt.show()

    def assortment_count(self):
        print(self.data.Assortment.value_counts())

    def assortment_share_pie(self):
        plt.figure(figsize=(12, 7))
        self.data.groupby('Assortment')['Sales'].sum().plot(kind='pie', autopct='%.2f%%')
        plt.title('Sales Share by Assortment Type', fontsize=15)
        plt.show()

    def average_sales_by_assortment(self):
        average_sales_by_assortment = self.data.groupby('Assortment')[
                                          'Sales'].sum() / self.data.Assortment.value_counts().sort_index()
        sns.barplot(x=average_sales_by_assortment.values, y=average_sales_by_assortment.index, orient='h')
        plt.xlabel('Sales', fontsize=15)
        plt.ylabel('Assortment', fontsize=15)
        plt.title('Sales Per Store by Assortment Type', fontsize=15)
        plt.show()

    def storetype_barplot(self, storetype, col):
        Type = self.data.loc[self.data.StoreType == storetype].set_index('Store')
        sns.barplot(y=Type[col].nlargest(10).index,
                    x=Type[col].nlargest(10).values,
                    order=Type[col].nlargest(10).index,
                    orient='h')

        plt.xlabel(col, fontsize=15)
        plt.ylabel('Store', fontsize=15)
        plt.title(f'Type{storetype.upper()} Store {col} Top 10', fontsize=15)
        plt.show()

    def storetype_hist(self, col):
        weight1 = np.ones_like(self.data.loc[self.data.StoreType == 'a'][col].index / len(
            self.data.loc[self.data.StoreType == 'a'][col].index))
        weight2 = np.ones_like(self.data.loc[self.data.StoreType == 'b'][col].index / len(
            self.data.loc[self.data.StoreType == 'b'][col].index))
        weight3 = np.ones_like(self.data.loc[self.data.StoreType == 'c'][col].index / len(
            self.data.loc[self.data.StoreType == 'c'][col].index))
        weight4 = np.ones_like(self.data.loc[self.data.StoreType == 'd'][col].index / len(
            self.data.loc[self.data.StoreType == 'd'][col].index))

        self.data.loc[self.data.StoreType == 'a'][col].hist(histtype='step', lw=2, weights=weight1)
        self.data.loc[self.data.StoreType == 'b'][col].hist(histtype='step', lw=2, weights=weight2)
        self.data.loc[self.data.StoreType == 'c'][col].hist(histtype='step', lw=2, weights=weight3)
        self.data.loc[self.data.StoreType == 'd'][col].hist(histtype='step', lw=2, weights=weight4)
        plt.xlabel(f'{col}', fontsize=15)
        plt.ylabel('Count', fontsize=15)
        plt.title(f'{col} Distribution by Store Type', fontsize=15)
        plt.legend(['A', 'B', 'C', 'D'])
        plt.show()

    def assortment_barplot(self, assortment, col):
        Type = self.data.loc[self.data.Assortment == assortment].set_index('Store')
        sns.barplot(y=Type[col].nlargest(10).index,
                    x=Type[col].nlargest(10).values,
                    order=Type[col].nlargest(10).index,
                    orient='h')

        plt.xlabel(col, fontsize=15)
        plt.ylabel('Store', fontsize=15)
        plt.title(f'Assortment {assortment.upper()}  {col} Top 10', fontsize=15)
        plt.show()

    def assortment_hist(self, col):
        weight1 = np.ones_like(self.data.loc[self.data.Assortment == 'a'][col].index / len(
            self.data.loc[self.data.Assortment == 'a'][col].index))
        weight2 = np.ones_like(self.data.loc[self.data.Assortment == 'b'][col].index / len(
            self.data.loc[self.data.Assortment == 'b'][col].index))
        weight3 = np.ones_like(self.data.loc[self.data.Assortment == 'c'][col].index / len(
            self.data.loc[self.data.Assortment == 'c'][col].index))
        self.data.loc[self.data.Assortment == 'a'][col].hist(histtype='step', lw=2, weights=weight1)
        self.data.loc[self.data.Assortment == 'b'][col].hist(histtype='step', lw=2, weights=weight2)
        self.data.loc[self.data.Assortment == 'c'][col].hist(histtype='step', lw=2, weights=weight3)
        plt.xlabel(f'{col}', fontsize=15)
        plt.ylabel('Count', fontsize=15)
        plt.title(f'Store {col} Distribution by Assortment Type', fontsize=15)
        plt.legend(['A', 'B', 'C'])
        plt.show()

    def type_violinplot(self, col, typecol):
        plt.figure(figsize=(12, 7))
        sns.violinplot(x=col, y=typecol,
                       data=self.data, order=sorted(set(self.data[typecol])),
                       orient='h')
        plt.title(f'{col} by {typecol}', fontsize=15)
        plt.xlabel(f'{col}', fontsize=15)
        plt.ylabel(f'{typecol}', fontsize=15)
        plt.show()

    def type_summary(self, typecol, col):
        return self.data.groupby(typecol)[col].describe().T

