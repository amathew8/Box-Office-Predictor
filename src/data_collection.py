#!/usr/bin/env python
# coding: utf-8

# In[390]:


import pandas as pd

from requests import get
from bs4 import BeautifulSoup
response = get('https://www.imdb.com/search/title/?title_type=feature&release_date=1975-01-01,2019-12-31&sort=boxoffice_gross_us,asc&count=250&ref_=adv_prv')
soup = BeautifulSoup(response.text, 'html.parser')
movie_list = soup.find_all('div',class_ = 'lister-item mode-advanced')

start = 0;
titles = []
years = []
genres = []
imdb_reviews = []
critic_reviews = []
mpaa_ratings = []
runtimes = []
directors = []
actors = []
domestic_grosses = []
budgets = []
production_companies = []


# In[391]:


for i in range(2):
    movie_list = []
    if i == 0:
        response = get('https://www.imdb.com/search/title/?title_type=feature&release_date=2020-01-01,&count=250')
        soup = BeautifulSoup(response.text, 'html.parser')
        movie_list = soup.find_all('div',class_ = 'lister-item mode-advanced')
        start = 251
    else:
        response = get(('https://www.imdb.com/search/title/?title_type=feature&release_date=2020-01-01,&count=250&start=251&ref_=adv_nxt'))
        soup = BeautifulSoup(response.text, 'html.parser')
        movie_list = soup.find_all('div',class_ = 'lister-item mode-advanced')
        start = start + 250
    for movie in movie_list:
        title = movie.h3.a.text
        titles.append(title)
        if movie.strong is not None:
            imdb_review = float(movie.strong.text)
            imdb_reviews.append(imdb_review)
        else:
            imdb_reviews.append("N/A")
        if movie.find('div', class_ = 'ratings-metascore') is not None:
            critic_review = movie.find('span',class_= 'metascore').text
            critic_reviews.append(critic_review)
        else:
            critic_reviews.append("N/A")
        year = movie.h3.find('span', class_ = 'lister-item-year').text
        years.append(year)
        if movie.find('span', class_ = 'certificate') is not None:
            mpaa_rating = movie.find('span', class_ = 'certificate').text
            mpaa_ratings.append(mpaa_rating)
        else:
            mpaa_ratings.append("N/A")
        if movie.find('span', class_ = 'genre') is not None:
            genre = movie.find('span', class_ = 'genre').text
            genres.append(genre.strip())
        else:
            genres.append("N/A")
        attributes = movie.find_all(attrs = {'name':'nv'})
        gross_added = False
        for attribute in attributes:
            if '$' in attribute.text:
                gross = attribute['data-value']
                domestic_grosses.append(gross)
                gross_added = True
                break
        if not gross_added:
            domestic_grosses.append('N/A')
        
        movie_tag = (movie.find('div', attrs = {'class':'ribbonize'})['data-tconst']) 
        response2 = get(('https://www.imdb.com/title/{}/?ref_=adv_li_tt'.format(movie_tag)))
        soup2 = BeautifulSoup(response2.text, 'html.parser')
        text_fields = soup2.find_all(class_ = "txt-block")
        cast_crew_fields = soup2.find_all(class_ = "credit_summary_item")
        prod_added = False
        budg_added = False
        director_added = False
        actors_added = False
        for text in text_fields:
            if 'Production Co:' in text.text:
                prod_companies = text.text.strip().split("\n")
                prod_companies.pop(0)
                prod_companies.pop(-1)
                production_companies.append(prod_companies)
                prod_added = True
            if 'Budget:' in text.text:
                budget = text.text.strip().split('\n')
                budget = (budget[0][8:])
                budgets.append(budget)
                budg_added = True
            if prod_added and budg_added:
                break
        for people in cast_crew_fields:
            if 'Director:' in people.text:
                director_array = (people.text.strip().split('\n'))
                directors.append(director_array[1])
                director_added = True
            if 'Stars:' in people.text:
                cast_string = ((people.text.strip().split('\n')) [1]).replace('|','')
                actors.append(cast_string.strip().split(','))
                actors_added = True
            if director_added and actors_added:
                break
        if not prod_added:
            production_companies.append('N/A')
        if not budg_added:
            budgets.append('N/A')
        if not director_added:
            directors.append('N/A')
        if not actors_added:
            actors.append('N/A')
         
 

    print(i)


# In[392]:


print(len(titles))
print(len(directors))
print(len(actors))
print(len(budgets))
print(len(domestic_grosses))
print(len(production_companies))


# In[393]:


movieDF2 = pd.DataFrame({'Movie': titles,
                      'Year':years,
                      'IMDB Review':imdb_reviews,
                      'Critic Review':critic_reviews,
                      'Genre(s)':genres,
                      'MPAA':mpaa_ratings,
                        'Director':directors,
                        'Box Office':domestic_grosses,
                        'Actors':actors,
                        'Prod. Company(s)':production_companies,
                        'Budget':budgets

})


# In[394]:


movieDF = pd.DataFrame({'Movie': titles,
                      'Year':years,
                      'IMDB Review':imdb_reviews,
                      'Critic Review':critic_reviews,
                      'Genre(s)':genres,
                      'MPAA':mpaa_ratings,
                        'Director':directors,
                        'Box Office':domestic_grosses,
                        'Actors':actors,
                        'Prod. Company(s)':production_companies,
                        'Budget':budgets

})


# In[ ]:





# In[395]:


movieDF2.to_excel("new_movies.xlsx")


# In[ ]:




