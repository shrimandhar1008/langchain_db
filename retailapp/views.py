from django.shortcuts import render
from langchain.llms import GooglePalm
from langchain.utilities import SQLDatabase
from langchain_experimental.sql import SQLDatabaseChain
from langchain.prompts import SemanticSimilarityExampleSelector
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.prompts import FewShotPromptTemplate
from langchain.chains.sql_database.prompt import PROMPT_SUFFIX, _mysql_prompt
from langchain.prompts.prompt import PromptTemplate
from . import few_shots
import os
from dotenv import load_dotenv
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt

load_dotenv()  # take environment variables from .env (especially openai api key)
embeddings = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')
to_vectorize = [" ".join(example.values()) for example in few_shots.few_shots]
vectorstore = Chroma.from_texts(to_vectorize, embeddings, metadatas=few_shots.few_shots)

# Database object and data
db_user = "root"
db_password = "Shri%400177"
db_host = "localhost"
db_name = "atliq_tshirts"
db = SQLDatabase.from_uri(f"mysql+pymysql://{db_user}:{db_password}@{db_host}/{db_name}",
                        sample_rows_in_table_info=3)
query_tshirt = 'SELECT * FROM atliq_tshirts.t_shirts;'
query_discounts = 'SELECT * FROM atliq_tshirts.discounts;'
query_tshirt_data = eval(db.run(query_tshirt))
query_discounts_data = db.run(query_discounts)
query_discounts_data = query_discounts_data.replace("Decimal","")
query_discounts_data = eval(query_discounts_data)

# Create your views here.
def show_index(request):
    return render(request,'index.html')


# @csrf_exempt
def get_few_shot_db_chain(request):
    if request.method == 'POST':
        user_question = request.POST.get('question')
        llm = GooglePalm(google_api_key=os.environ["GOOGLE_API_KEY"], temperature=0.1)
        example_selector = SemanticSimilarityExampleSelector(
            vectorstore=vectorstore,
            k=2,
        )
        example_prompt = PromptTemplate(
            input_variables=["Question", "SQLQuery", "SQLResult","Answer",],
            template="\nQuestion: {Question}\nSQLQuery: {SQLQuery}\nSQLResult: {SQLResult}\nAnswer: {Answer}",
        )
        few_shot_prompt = FewShotPromptTemplate(
            example_selector=example_selector,
            example_prompt=example_prompt,
            prefix=_mysql_prompt,
            suffix=PROMPT_SUFFIX,
            input_variables=["input", "table_info", "top_k"], #These variables are used in the prefix and suffix
        )
        chain = SQLDatabaseChain.from_llm(llm, db, verbose=True, return_intermediate_steps=True, prompt=few_shot_prompt)
        answer = chain(user_question)
        result = answer['result']
        returned_query = answer['intermediate_steps'][2]['sql_cmd']
        data = {'answer':result,'tshirt_dict':query_tshirt_data,'discounts_data_dict':query_discounts_data,'question':user_question, 'returned_query':returned_query}
        return JsonResponse(data,  content_type="application/json" ,safe=False)   