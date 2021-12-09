python scripts/run_vae.py \
-m model_name=NeuralSinkhornTopicModel,WassersteinLatentDirichletAllocation,NeuralVariationalDocumentModel,ProductOfExpertsLatentDirichletAllocation,NeuralVariationalLatentDirichletAllocation,GaussianSoftmaxModel,GaussianStickBreakingModel,RecurrentStickBreakingModel \
corpus_name=StackOverflow,SearchSnippets,TrecTweet,GoogleNews,PascalFlicker,20NewsGroups,BBCNews,DBLP,M10 \
model.lr=0.005 model.lr_dec=0.001 \
model.batch_size=256 model.n_components=50 model.max_epochs=200 \
use_pwe=$1 train_pwe=$2