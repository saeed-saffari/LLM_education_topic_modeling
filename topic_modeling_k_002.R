# Install and load necessary packages (only run install.packages() if not installed)
#install.packages(c("tm", "stm", "textstem", "tidyverse"))
library(tm)
library(stm)
library(textstem)
library(tidyverse)

# Step 1: Load and Inspect the Data
data <- read.csv("/Users/saeed/Library/CloudStorage/OneDrive-DalhousieUniversity/0- Dalhousie University - Saeed/4- Fall 2024/RA - SMVU/Topic modeling/Tastic AI - Opensource - AI tools.csv")
#head(data$desc)

# Step 2: Preprocess the Text Data (use VCorpus instead of SimpleCorpus)
desc_corpus <- VCorpus(VectorSource(data$desc))
desc_corpus <- tm_map(desc_corpus, content_transformer(tolower))
desc_corpus <- tm_map(desc_corpus, removePunctuation)
desc_corpus <- tm_map(desc_corpus, removeNumbers)
desc_corpus <- tm_map(desc_corpus, removeWords, stopwords("en"))
desc_corpus <- tm_map(desc_corpus, content_transformer(lemmatize_strings))

# Step 3: Convert to Document-Term Matrix (DTM)
desc_dtm <- DocumentTermMatrix(desc_corpus)

# Convert the DTM to STM-compatible format
#docs <- apply(desc_dtm, 1, function(x) {
#  words <- as.integer(x[x > 0])
#  terms <- as.integer(names(x[x > 0]))
#  list(words = words, terms = terms)
#})

# Convert the DTM to STM-compatible format
# Convert DTM to STM format, ensuring no duplicate terms within each document
docs <- lapply(1:nrow(desc_dtm), function(i) {
  term_indices <- desc_dtm[i, ]$i
  term_counts <- desc_dtm[i, ]$v
  
  # Summing counts for any duplicate terms within a document
  if (length(term_indices) > 0) {
    term_df <- data.frame(term_indices, term_counts)
    term_df <- aggregate(term_counts ~ term_indices, data = term_df, FUN = sum)
    
    # Convert to matrix format as required by STM: two rows (terms, counts)
    return(rbind(as.integer(term_df$term_indices), as.integer(term_df$term_counts)))
  } else {
    return(NULL)
  }
})


# Remove NULL entries from docs to avoid empty documents
docs <- Filter(Negate(is.null), docs)

vocab <- Terms(desc_dtm)


# Step 4: Prepare Data for STM
# Structure the data as required by STM (a list with documents and vocabulary)
desc_stm <- list(documents = docs, vocab = vocab)
library(stm)


# Step 5: Determine Optimal Number of Topics
#topic_range <- seq(5, 20, by = 5)  # Adjust the range as needed
vocab_size <- length(vocab)
topic_range <- seq(2, min(20, vocab_size - 1), by = 2)  # Example, adjust as needed
#models <- searchK(docs, vocab, K = topic_range)
models <- searchK(docs, vocab, K = topic_range, init.type = "LDA")
# Plot the model quality metrics to decide on the optimal topic number
plot(models)

# Choose an optimal number of topics (for example, based on semantic coherence)
optimal_k <- models$K[which.max(models$semcoh)]  # Example: based on semantic coherence

# Step 6: Fit the STM Model with the Optimal Number of Topics
stm_model <- stm(documents = docs, vocab = vocab, K = optimal_k, max.em.its = 75)

# Step 7: Interpret the Topics
# FREX Words: Use FREX to find the top words for each topic
topics <- labelTopics(stm_model, n = 10)
print(topics)

# Step 8: Visualize Topic Prevalence (optional)
plot(stm_model, type = "summary")

# Step 9: Additional Visualizations and Model Saving
# Topic correlation network
topicCorr(stm_model) %>% plot

# Save the Model for Future Use
saveRDS(stm_model, file = "stm_model.rds")

# To load this model again:
# stm_model <- readRDS("stm_model.rds")







desc_corpus <- VCorpus(VectorSource(data$desc))
desc_corpus <- tm_map(desc_corpus, content_transformer(tolower))
desc_corpus <- tm_map(desc_corpus, removePunctuation)
desc_corpus <- tm_map(desc_corpus, removeNumbers)
desc_corpus <- tm_map(desc_corpus, removeWords, stopwords("en"))
desc_corpus <- tm_map(desc_corpus, content_transformer(lemmatize_strings))

# Step 2: Create Document-Term Matrix (DTM)
desc_dtm <- DocumentTermMatrix(desc_corpus)

# Step 3: Extract `docs` and `vocab` directly from the DTM
# `desc_dtm` provides both the vocabulary terms and term counts
dtm_matrix <- as.matrix(desc_dtm)

# Define `vocab` directly from the DTM's terms
vocab <- colnames(dtm_matrix)

# Define `docs` by iterating through each document and only including non-zero counts
docs <- lapply(1:nrow(dtm_matrix), function(i) {
  term_indices <- which(dtm_matrix[i, ] > 0)  # Find terms with non-zero counts
  term_counts <- dtm_matrix[i, term_indices]  # Get counts for those terms
  
  if (length(term_indices) > 0) {
    # Construct a two-row integer matrix for each document
    return(matrix(as.integer(c(term_indices, term_counts)), nrow = 2, byrow = TRUE))
  } else {
    return(NULL)
  }
})

# Remove empty documents from `docs`
docs <- Filter(Negate(is.null), docs)


test_model <- stm(docs, vocab, K = 5, init.type = "LDA", max.em.its = 50)

summary(test_model)

topic_range <- seq(5, 20, by = 5)
models <- searchK(docs, vocab, K = topic_range)
plot(models)




education_data <- data %>% filter(str_detect(cat, regex("AI tools for education", ignore_case = TRUE)))

descriptions <- education_data$desc


# Create a text corpus and preprocess as before
desc_corpus <- VCorpus(VectorSource(descriptions))
desc_corpus <- tm_map(desc_corpus, content_transformer(tolower))
desc_corpus <- tm_map(desc_corpus, removePunctuation)
desc_corpus <- tm_map(desc_corpus, removeNumbers)
desc_corpus <- tm_map(desc_corpus, removeWords, stopwords("en"))
desc_corpus <- tm_map(desc_corpus, content_transformer(lemmatize_strings))

# Step 3: Create Document-Term Matrix (DTM)
desc_dtm <- DocumentTermMatrix(desc_corpus)

# Step 4: Extract `docs` and `vocab`
# Convert DTM to matrix format and extract terms and counts
dtm_matrix <- as.matrix(desc_dtm)

# Define `vocab` directly from the DTM's terms
vocab <- colnames(dtm_matrix)

# Define `docs` by iterating through each document and only including non-zero counts
docs <- lapply(1:nrow(dtm_matrix), function(i) {
  term_indices <- which(dtm_matrix[i, ] > 0)  # Find terms with non-zero counts
  term_counts <- dtm_matrix[i, term_indices]  # Get counts for those terms
  
  if (length(term_indices) > 0) {
    # Construct a two-row integer matrix for each document
    return(matrix(as.integer(c(term_indices, term_counts)), nrow = 2, byrow = TRUE))
  } else {
    return(NULL)
  }
})

# Remove empty documents from `docs`
docs <- Filter(Negate(is.null), docs)

# Step 5: Run searchK to Determine the Optimal Number of Topics
topic_range <- seq(5, 26, by = 1)  # Adjust range as needed
models <- searchK(docs, vocab, K = topic_range)
plot(models)





optimal_k <- 15 #models$K[which.max(models$semcoh)]  # Select based on highest coherence
final_model <- stm(docs, vocab, K = optimal_k, init.type = "LDA", max.em.its = 50)

summary(final_model)

# Step 2: Extract the Top Words for Each Topic
topic_labels <- labelTopics(final_model, n = 10)  # Extract top 10 words per topic
print(topic_labels)






education_keywords <- c("teach", "teaching", "learn", "learning", "classroom", "student",
                        "rephrase", "plagiarism", "instructor", "curriculum", "assessment", 
                        "homework", "school", "educate", "education", "tutor")

# Function to check if text contains any education-related keywords
contains_education_terms <- function(text, keywords) {
  any(sapply(keywords, function(kw) grepl(kw, text, ignore.case = TRUE)))
}

# Filter descriptions that mention any educational keywords
filtered_data <- data[sapply(data$desc, contains_education_terms, education_keywords), ]

# Extract filtered descriptions for topic modeling
descriptions <- filtered_data$desc

desc_corpus <- VCorpus(VectorSource(descriptions))
desc_corpus <- tm_map(desc_corpus, content_transformer(tolower))
desc_corpus <- tm_map(desc_corpus, removePunctuation)
desc_corpus <- tm_map(desc_corpus, removeNumbers)
desc_corpus <- tm_map(desc_corpus, removeWords, stopwords("en"))
desc_corpus <- tm_map(desc_corpus, content_transformer(lemmatize_strings))

# Create Document-Term Matrix (DTM) for the filtered data
desc_dtm <- DocumentTermMatrix(desc_corpus)

# Extract `docs` and `vocab` for STM
dtm_matrix <- as.matrix(desc_dtm)
vocab <- colnames(dtm_matrix)
docs <- lapply(1:nrow(dtm_matrix), function(i) {
  term_indices <- which(dtm_matrix[i, ] > 0)
  term_counts <- dtm_matrix[i, term_indices]
  
  if (length(term_indices) > 0) {
    return(matrix(as.integer(c(term_indices, term_counts)), nrow = 2, byrow = TRUE))
  } else {
    return(NULL)
  }
})
docs <- Filter(Negate(is.null), docs)

# Step: Use searchK to find the optimal number of topics
library(stm)
topic_range <- seq(5, 26, by = 2)  # Adjust range as needed
models <- searchK(docs, vocab, K = topic_range)
plot(models)





# ------------------------------------




# Define keywords for filtering
education_keywords <- c("teach", "teaching", "learn", "learning", "classroom", "student",
                        "rephrase", "plagiarism", "instructor", "curriculum", "assessment", 
                        "homework", "school", "educate", "education", "tutor")

exclusion_keywords <- c("advertising", "branding", "budgeting", "client", "CRM", "customer",
                        "e-commerce", "expense", "finance", "investment", "logo", "marketing",
                        "operations", "pipeline", "product", "profit", "resource allocation", 
                        "sales", "SEO")

# Step 1: Filter for Education-Related Terms
contains_education_terms <- function(text, keywords) {
  any(sapply(keywords, function(kw) grepl(kw, text, ignore.case = TRUE)))
}

education_filtered_data <- data[sapply(data$desc, contains_education_terms, education_keywords), ]

# Step 2: Exclude Rows with Business Terms
contains_exclusion_terms <- function(text, keywords) {
  any(sapply(keywords, function(kw) grepl(kw, text, ignore.case = TRUE)))
}

# Apply exclusion filter to remove rows with business-related terms
final_filtered_data <- education_filtered_data[!sapply(education_filtered_data$desc, contains_exclusion_terms, exclusion_keywords), ]

# Extract the final descriptions for topic modeling
descriptions <- final_filtered_data$desc


desc_corpus <- VCorpus(VectorSource(descriptions))
desc_corpus <- tm_map(desc_corpus, content_transformer(tolower))
desc_corpus <- tm_map(desc_corpus, removePunctuation)
desc_corpus <- tm_map(desc_corpus, removeNumbers)
desc_corpus <- tm_map(desc_corpus, removeWords, stopwords("en"))
desc_corpus <- tm_map(desc_corpus, content_transformer(lemmatize_strings))

# Create Document-Term Matrix (DTM) for the filtered data
desc_dtm <- DocumentTermMatrix(desc_corpus)

# Extract `docs` and `vocab` for STM
dtm_matrix <- as.matrix(desc_dtm)
vocab <- colnames(dtm_matrix)
docs <- lapply(1:nrow(dtm_matrix), function(i) {
  term_indices <- which(dtm_matrix[i, ] > 0)
  term_counts <- dtm_matrix[i, term_indices]
  
  if (length(term_indices) > 0) {
    return(matrix(as.integer(c(term_indices, term_counts)), nrow = 2, byrow = TRUE))
  } else {
    return(NULL)
  }
})
docs <- Filter(Negate(is.null), docs)

# Step: Use searchK to find the optimal number of topics
library(stm)
topic_range <- seq(5, 26, by = 1)  # Adjust range as needed
models <- searchK(docs, vocab, K = topic_range)
plot(models)

final_model <- stm(docs, vocab, K = 15, init.type = "LDA", max.em.its = 75)

# View a summary of the model
summary(final_model)


# Extract and print the top words for each topic
topic_labels <- labelTopics(final_model, n = 10)  # Top 10 words per topic
print(topic_labels)


for (i in 1:15) {
  cat(paste("Topic", i, "Top Words:", paste(topic_labels$frex[i, ], collapse = ", ")), "\n")
}












