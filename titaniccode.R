# Load packages
library('ggplot2') # visualization
library('ggthemes') # visualization
library('scales') # visualization
library('dplyr') # data manipulation
library('mice') # imputation
library('randomForest') # classification algorithm
train <- read.csv(?C:\\Users\\Rebecca\\Desktop\\machinelearning\\titanic\\train.csv', stringsAsFactors = F)
test <- read.csv('C:\\Users\\Rebecca\\Desktop\\machinelearning\\titanic\\test.csv', stringsAsFactors = F)

full  <- bind_rows(train, test) # bind training & test data
?# check data
str(full)

# Grab title from passenger names
full$Title <- gsub('(.*, )|(\\..*)', '', full$Name)

# Show title counts by sex
table(full$Sex, full$Title)
# Titles with very low cell counts to be combined to "rare" level
rare_title <- c('Dona', ?Lady', 'the Countess','Capt', 'Col', 'Don', 
                'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer')

# Also reassign mlle, ms, and mme accordingly
full$Title[full$Title == 'Mlle']        <- 'Miss' 
full$Title[full$Title == 'Ms']          <- 'Miss'
full$T?tle[full$Title == 'Mme']         <- 'Mrs' 
full$Title[full$Title %in% rare_title]  <- 'Rare Title'

# Show title counts by sex again
table(full$Sex, full$Title)
# Finally, grab surname from passenger name
full$Surname <- sapply(full$Name,  
               ?       function(x) strsplit(x, split = '[,.]')[[1]][1])

cat(paste('We have <b>', nlevels(factor(full$Surname)), 
          '</b> unique surnames. I would be interested to infer ethnicity based on surname --- another time.'))

# Create a family size variab?e including the passenger themselves
full$Fsize <- full$SibSp + full$Parch + 1
# Create a family variable 
full$Family <- paste(full$Surname, full$Fsize, sep='_')
# Use ggplot2 to visualize the relationship between family size & survival: singletons and la?ge families die. small 
#families survive
ggplot(full[1:891,], aes(x = Fsize, fill = factor(Survived))) +
  geom_bar(stat='count', position='dodge') +
  scale_x_continuous(breaks=c(1:11)) +
  labs(x = 'Family Size') +
  theme_few()

# Discretize family siz?
full$FsizeD[full$Fsize == 1] <- 'singleton'
full$FsizeD[full$Fsize < 5 & full$Fsize > 1] <- 'small'
full$FsizeD[full$Fsize > 4] <- 'large'



#Checking missing values in existing features
sum(is.na(full$PassengerId)) #0
sum(is.na(full$Survived)) #418  of ?ourse, it's the test set without predictions
sum(is.na(full$Pclass))#0
sum(is.na(full$Name)) #0
sum(is.na(full$Sex)) #0
sum(is.na(full$Age))#263  maybe we can put the mean
sum(is.na(full$SibSp))#0
sum(is.na(full$Parch))#0
sum(is.na(full$Ticket))#0
sum(is.n?(full$Fare))#1   #put the mean of its sector
sum(is.na(full$Cabin))#0
#BUT
summary(full$Cabin)
#Length     Class      Mode 
#1309 character character 
#sum(full$Cabin == "")
#[1] 1014  SO the vector is 1309 rows long and 1014 don't have values
sum(is.na(fu?l$Embarked))#0
#sum(full$Embarked == "")  so embarked has two MV
#created ones don't have MV
sum(is.na(full$Title))#0
sum(is.na(full$Surname))#0
sum(is.na(full$FSize))#0
sum(is.na(full$Family))#0
sum(is.na(full$FsizeD))#0

# This variable appears to have a?lot of missing values
full$Cabin[1:28]
# The first character is the deck. For example:
strsplit(full$Cabin[2], NULL)[[1]]
# Create a Deck variable. Get passenger deck A - F:
full$Deck<-factor(sapply(full$Cabin, function(x) strsplit(x, NULL)[[1]][1]))

# Us? ggplot2 to visualize the relationship between deck & survival:B,C,D,E,F survive. A,(G),T,NA die
ggplot(full[1:891,], aes(x = Deck, fill = factor(Survived))) +
  geom_bar(stat='count', position='dodge') +
  labs(x = 'deck') +
  theme_few()

#small dataset ?o no deleting rows. Age we can put the mean/median, deck we can see the fare

# Passengers 62 and 830 are missing Embarkment
full[c(62, 830), 'Embarked']
cat(paste('We will infer their values for **embarkment** based on present data that we can imagine may?be relevant: **passenger class** and **fare**. We see that they paid<b> $', full[c(62, 830), 'Fare'][[1]][1], '</b>and<b> $', full[c(62, 830), 'Fare'][[1]][2], '</b>respectively and their classes are<b>', full[c(62, 830), 'Pclass'][[1]][1], '</b>and<b>', f?ll[c(62, 830), 'Pclass'][[1]][2], '</b>. So from where did they embark?'))
# Get rid of our missing passenger IDs
embark_fare <- full %>%
  filter(PassengerId != 62 & PassengerId != 830)
# Use ggplot2 to visualize embarkment, passenger class, & median fare?ggplot(embark_fare, aes(x = Embarked, y = Fare, fill = factor(Pclass))) +
  geom_boxplot() +
  geom_hline(aes(yintercept=80), 
             colour='red', linetype='dashed', lwd=2) +
  scale_y_continuous(labels=dollar_format()) +
  theme_few()
#The median f?re for a first class passenger departing from Charbourg ('C') coincides nicely with the $80 paid therefore 
#we replace NA with C# Since their fare was $80 for 1st class, they most likely embarked from 'C'
ggplot(data=full, aes(x=Embarked, y=Fare, fill=fac?or(Pclass))) +
  geom_bar(stat="identity", position=position_dodge())+
  geom_text(aes(label=Fare), vjust=1.6, color="white",  
            position = position_dodge(0.9), size=3.5)+ 
  scale_fill_brewer(palette="Paired")+
  theme_minimal()



ggplot(full,?aes(x=Embarked, y=Fare, fill=factor(Pclass))) + 
  geom_bar(stat="identity", position = "dodge") + 
  scale_fill_brewer(palette = "Set1")

# Since their fare was $80 for 1st class, they most likely embarked from 'C'
full$Embarked[c(62, 830)] <- 'C'

fisher?test( table(full$Embarked, full$Fare),simulate.p.value = TRUE )

#Fisher's Exact Test for Count Data with simulated p-value (based on 2000 replicates)

#data:  table(full$Embarked, full$Fare)
#p-value = 0.0004998
#alternative hypothesis: two.sided
# two fa?tors are not independent

newAge<- full$Age[which(full$Age!="NA")]  #non NA values

h <- hist(newAge, breaks = 10, density = 10,
          col = "lightgray", xlab = "Age", main = "Age distribution") 
xfit <- seq(min(newAge), max(newAge), length = 40) 
yfit?<- dnorm(xfit, mean = mean(newAge), sd = sd(newAge)) 
yfit <- yfit * diff(h$mids[1:2]) * length(newAge) 
lines(xfit, yfit, col = "black", lwd = 2)
#so it's a gaussian and we can replace with the mean


h<-hist(newAge, density=20, breaks=20, prob=TRUE, 
   ? xlab="newAge", 
     main="Age distribution")
xfit <- seq(min(newAge), max(newAge), length = 40) 
yfit <- dnorm(xfit, mean = mean(newAge), sd = sd(newAge))
lines(xfit, yfit, col = "darkblue", lwd = 2)
#it looks a bit skewwd...

library(ggpubr)
ggqqplot(ne?Age)
shapiro.test(newAge)
#Null hypothesis: The data is normally distributed. If p> 0.05, normality can be assumed
#W = 0.97955, p-value = 5.747e-11  p value<<<0.05 so we ca
#suggesting strong evidence of non-normality. So maybe we could replace missing va?ues with the median since it's a skewed distribution


#BACK TO MISSING VALUES:FARE
# Show row 1044
full[1044, ] #it has no survived prediction, no fare, no cabin (no deck)
#Let's visualize Fares among all others sharing their class and embarkment (n = 494?
ggplot(full[full$Pclass == '3' & full$Embarked == 'S', ], 
       aes(x = Fare)) +
  geom_density(fill = '#99d6ff', alpha=0.4) + 
  geom_vline(aes(xintercept=median(Fare, na.rm=T)),
             colour='red', linetype='dashed', lwd=1) +
  geom_vline(aes(x?ntercept=mean(Fare, na.rm=T)),
             colour='black', linetype='dashed', lwd=1)+
  scale_x_continuous(labels=dollar_format()) +
  theme_few()
#red is median, black is mean. we should replace it with the median since it's strongly skewed
# Replace mis?ing fare value with median fare for class/embarkment
full$Fare[1044] <- median(full[full$Pclass == '3' & full$Embarked == 'S', ]$Fare, na.rm = TRUE)
#now missing values for Age
# Show number of missing Age values
sum(is.na(full$Age))

library(mice)
# Make ?ariables factors into factors
factor_vars <- c('Pclass','Sex','Embarked',
                 'Title','Surname','Family','FsizeD')

full[factor_vars] <- lapply(full[factor_vars], function(x) as.factor(x))

# Set a random seed
set.seed(130)

# Perform mice imp?tation, excluding certain less-than-useful variables:
mice_mod <- mice(full[, !names(full) %in% c('PassengerId','Name','Ticket','Cabin','Family','Surname','Survived')], method='rf') 

# Save the complete output 
mice_output <- mice::complete(mice_mod)

# P?ot age distributions
par(mfrow=c(1,2))
hist(full$Age, freq=F, main='Age: Original Data', 
     col='darkgreen', ylim=c(0,0.04))
hist(mice_output$Age, freq=F, main='Age: MICE Output', 
     col='lightgreen', ylim=c(0,0.04))

# Replace Age variable from the ?ice model.
full$Age <- mice_output$Age

# Show new number of missing Age values
sum(is.na(full$Age))

#create age group column
full$age_group[full$Age<18]<-'1'
full$age_group[full$Age>17]<-'2'
full$age_group[full$Age>55]<-'3'
#create child, mother
# First ?e'll look at the relationship between age & survival
ggplot(full[1:891,], aes(Age, fill = factor(Survived))) + 
  geom_histogram() + 
  # I include Sex since we know (a priori) it's a significant predictor
  facet_grid(.~Sex) + 
  theme_few()

# Create the?column child, and indicate whether child or adult
full$Child[full$Age < 18] <- 'Child'
full$Child[full$Age >= 18] <- 'Adult'

# Show counts
table(full$Child, full$Survived)

# Adding Mother variable
full$Mother <- 'Not Mother'
full$Mother[full$Sex == 'fema?e' & full$Parch > 0 & full$Age > 18 & full$Title != 'Miss'] <- 'Mother'

# Show counts
table(full$Mother, full$Survived)
# Finish by factorizing our two new factor variables
full$Child  <- factor(full$Child)
full$Mother <- factor(full$Mother)
md.pattern(fu?l)


set.seed(754)



#still dome analysis
# Use ggplot2 to visualize the relationship between pclass and survival
ggplot(full[1:891,], aes(x = Pclass, fill = factor(Survived))) +
  geom_bar(stat='count', position='dodge') +
  labs(x = 'Pclass') +
  theme_?ew()
#that Pclass is contributing to a persons chance of survival, especially if this person is in class 1. 
#showing the number of people for each class in each age group
ggplot(full[1:891,], aes(x = age_group, fill = factor(Pclass))) +
  geom_bar(positio?='dodge')+
  labs(x = 'Age group') +
  theme_few()


ggplot(full[1:891,], aes(x = age_group, fill = factor(Survived))) +
  geom_bar(position='dodge')+
  labs(x = 'Age group') +
  theme_few()
#so age goup made a huge difference mainly for adults

ggplot(ful?[1:891,], aes(x=age_group,y = Survived), fill = factor(full$Pclass)) +
  labs(x = 'Age group') +
  theme_few()
ggplot(full[1:891,], aes(x=age_group,y = Survived, fill = factor(Pclass))) +geom_bar(stat = "identity")+
  theme_few()
#drop ticket column
#drop ?ge column


#let's try to find cabins based on fares

library(mice)
# Make variables factors into factors
factor_variables<- c('Pclass','Sex','Embarked',
                 'Title','Surname','Family','FsizeD','Child','Mother','age_group')

full[factor_variab?es] <- lapply(full[factor_variables], function(x) as.factor(x))

# Set a random seed
set.seed(130)

# Perform mice imputation, excluding certain less-than-useful variables:
mice_module <- mice(full[, !names(full) %in% c('Name','Sex','age_group')], method='?f') 

# Save the complete output 
mice_outputtwo <- mice::complete(mice_module)

#plot two distributions
p1<-ggplot(full[1:891,], aes(x = Deck), colour="#000099") +
  geom_bar(position='dodge', fill="#FF9999")+
  labs(x = 'Deck',title="Original Data")+
  t?eme_few()
p2<-ggplot(mice_outputtwo[1:891,], aes(x = Deck)) +
  geom_bar(position='dodge', fill="#000099", colour="black")+
  labs(x = 'Deck', title="Imputation data") +
  theme_few()
grid.arrange(p1, p2, nrow = 1)


# Replace deck variable from the mice m?del.
full$Deck <- mice_output$Deck

# Set a random seed
set.seed(754)



#create new column age*class
full$age_times_class<-as.numeric(full$Age)*as.numeric(full$Pclass)

#lets drop name, we just needed it to get the title
full$Name<-NULL
#change name
colna?es(full)[colnames(full)=="FSizeD"] <- "family_size_class"
colnames(full)[colnames(full)=="Fsize"] <- "family_size"



full$PassengerId<-NULL #cause it's completely unuseful
#lets drop cabin
full$Cabin<-NULL
#same for ticket
full$Ticket<-NULL
#same for fare?full$Fare<-NULL
################################RANDOM FOREST
full$age_times_classD[full$age_times_class<25]<-'1'
full$age_times_classD[full$age_times_class<50& full$age_times_class>25]<-'2'
full$age_times_classD[full$age_times_class>50]<-'3'
full$age_time?_class<-NULL
full$age_times_classD[full$age_times_classD=='1']<-'A'
full$age_times_classD[full$age_times_classD=='2']<-'B'
full$age_times_classD[full$age_times_classD=='3']<-'C'
# Split the data back into a train set and a test set
train <- full[1:891,]
te?t <- full[892:1309,]
# Build the model (note: not all possible variables are used)
rf_model <- randomForest(factor(Survived) ~ Pclass + Sex + SibSp + Parch 
                            + Embarked + Title + Deck+
                           FsizeD + Child + ?other+age_group,
                         data = train)

# Show model error
plot(rf_model, ylim=c(0,0.36))
legend('topright', colnames(rf_model$err.rate), col=1:3, fill=1:3)


# Get importance
importance    <- importance(rf_model)
varImportance <- data.fra?e(Variables = row.names(importance), 
                            Importance = round(importance[ ,'MeanDecreaseGini'],2))

# Create a rank variable based on importance
rankImportance <- varImportance %>%
  mutate(Rank = paste0('#',dense_rank(desc(Importanc?))))

# Use ggplot2 to visualize the relative importance of variables
ggplot(rankImportance, aes(x = reorder(Variables, Importance), 
                           y = Importance, fill = Importance)) +
  geom_bar(stat='identity') + 
  geom_text(aes(x = Variab?es, y = 0.5, label = Rank),
            hjust=0, vjust=0.55, size = 4, colour = 'red') +
  labs(x = 'Variables') +
  coord_flip() + 
  theme_few()
# Predict using the test set
prediction <- predict(rf_model, test)

# Save the solution to a dataframe with t?o columns: PassengerId and Survived (prediction)
solution <- data.frame(PassengerID = test$PassengerId, Survived = prediction)

# Write the solution to file
write.csv(solution, file = 'rf_mod_Solution.csv', row.names = F)



#Confusion matrix:
#0   1 class?error
#0 499  50  0.09107468
#1 112 230  0.32748538

# 0.8181818

#try with different features
rf_modeltwo <- randomForest(factor(Survived) ~ Title + Sex + Pclass+ Deck + 
                           FsizeD,
                         data = train)
# Show mod?l error
plot(rf_modeltwo, ylim=c(0,0.36))
legend('topright', colnames(rf_model$err.rate), col=1:3, fill=1:3)


# Get importance
importancetwo   <- importance(rf_modeltwo)
varImportancetwo <- data.frame(Variablestwo = row.names(importancetwo), 
            ?               Importancetwo = round(importancetwo[ ,'MeanDecreaseGini'],2))

# Create a rank variable based on importance
rankImportancetwo <- varImportancetwo %>%
  mutate(Rank = paste0('#',dense_rank(desc(Importancetwo))))

# Use ggplot2 to visualize th? relative importance of variables
ggplot(rankImportancetwo, aes(x = reorder(Variablestwo, Importancetwo), 
                           y = Importancetwo, fill = Importancetwo)) +
  geom_bar(stat='identity') + 
  geom_text(aes(x = Variablestwo, y = 0.5, labe? = Rank),
            hjust=0, vjust=0.55, size = 4, colour = 'red') +
  labs(x = 'Variables') +
  coord_flip() + 
  theme_few()


#Confusion matrix:
#0   1 class.error
#0 492  57   0.1038251
#1  96 246   0.2807018


#0.8282828
######################




#?1 is not an extremely high number of variables, but still we can drop the last 6 and get an accuracy of 0.8462402
#against the  0.8372615 of before. we drop them cause we don't want to overfit


#i was thinking of adding the distance between cabins and boa?s but basically they go in order like A,B ae first class 
#so they're higher while the others are lower so it took more time to them to escape

