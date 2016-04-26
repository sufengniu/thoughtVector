

echo "Downloading..."

wget --no-check-certificate http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz

echo "unzuipping..."

tar xvf aclImdb_v1.tar.gz
mv aclImdb data

echo "Done ."
