# CycleGANs - Image Translation Project

A comprehensive machine learning project implementing CycleGANs for unpaired image-to-image translation, developed under the [IITK Consulting Group](https://iitkconsult.org/) of Science and Technology Council, IIT Kanpur during May'24-Jul'24.

## Project Overview

CycleGANs (Cycle-Consistent Generative Adversarial Networks) represent a breakthrough in image-to-image translation tasks. This project demonstrates high-quality image translation without requiring paired datasets, making it extremely versatile for applications like style transfer, photo enhancement, and domain adaptation.

The project follows a structured 8-week learning path, progressing from fundamental machine learning concepts to advanced generative models, culminating in the implementation of CycleGANs for translating summer landscapes to winter landscapes and vice versa.

## Key Features

- **Unpaired Image Translation**: Performs image-to-image translation without requiring corresponding image pairs
- **Multiple GAN Architectures**: Implements DCGAN, Conditional GAN, Pix2Pix, and CycleGAN
- **Advanced Loss Functions**: Utilizes adversarial loss, cycle consistency loss, and identity loss
- **Comprehensive Pipeline**: Complete preprocessing, training, and evaluation workflow

## Project Timeline and Implementation

### Week 1: Python Foundations
- Python syntax and programming fundamentals
- NumPy and Pandas for data manipulation
- Setting up the development environment

**Assignment:** Python programming and data science basics
- **Core Programming**: Python syntax mastery and efficient coding practices
- **NumPy Fundamentals**: Array operations, mathematical computations, and numerical methods
- **Pandas Basics**: Data manipulation, cleaning, and analysis techniques
- **Objective**: Establish strong foundation in Python ecosystem for machine learning applications

### Week 2: Machine Learning Basics
- Linear and logistic regression implementation
- K-means clustering algorithm
- Principal Component Analysis (PCA)
- Regularization techniques (L1, L2)
- Gradient descent and Newton-Raphson method

**Assignment:** Comprehensive ML fundamentals implementation
- **From-Scratch Implementations**: Linear and logistic regression without libraries
- **Library-Based Implementations**: Same algorithms using scikit-learn for comparison
- **Advanced Topics**: K-means clustering, closed-form solutions, Newton-Raphson method
- **Evaluation Metrics**: MSE, accuracy, and comprehensive model assessment
- **Key Components**: Multiple linear regression, data normalization, and optimization techniques

### Week 3: Neural Networks
- Neural network implementation from scratch
- Backpropagation algorithm
- Optimizers (Adam, SGD, mini-batch gradient descent)
- Regularization techniques (dropout)
- TensorFlow framework integration

**Assignment:** Three-part comprehensive assignment covering:
1. **Theoretical Foundation**: Deep learning concepts and neural network fundamentals
2. **From-Scratch Implementation**: Multiclass classification neural network built without libraries
3. **Framework Implementation**: Data preprocessing, visualization, and TensorFlow-based training

**Offline Test:** Feature prediction using neural networks to validate conceptual understanding

### Week 4: Convolutional Neural Networks
- CNN architecture and components
- Convolution and pooling layers
- Data augmentation techniques
- Transfer learning implementation
- Image classification tasks

**Assignment:** Two-part practical implementation:
1. **Core Implementation**: Custom convolution filters and pooling layers from scratch
2. **Advanced Applications**: Data augmentation visualization, CNN model training, and transfer learning evaluation across multiple architectures

**Major Assessment - True Triumph 2 (100 marks)**
- **Duration**: 1.5 hours
- **Task**: Plant disease classification using CNN on a 3-class plant leaf dataset
- **Challenge**: Train model on provided training data and evaluate on test set with classification report
- **Constraints**: No AI assistance (ChatGPT/Colab/Kaggle), Google research allowed
- **Deliverables**: 
  - Classification scores recorded in attendance sheet
  - Screenshot submissions by 8:00 PM
  - Python code upload to GitHub test folder by 8:30 PM
- **Assessment**: Graded evaluation testing practical CNN implementation skills

### Week 5: Introduction to GANs - DCGAN
- Generative Adversarial Networks fundamentals
- DCGAN architecture and training
- Generator and discriminator competition
- Image generation for English alphabet dataset

**Assignment:** English alphabet generation using GANs
- **Dataset**: Typed letters in various fonts across English alphabet
- **Preprocessing**: Image normalization and standardization
- **Training**: Approximately 1000 epochs with adversarial training
- **Output**: Trained model saved as .h5 file for letter image generation
- **Objective**: Generate realistic English letter images across different font styles

### Week 6: Conditional GANs
- Conditional GAN architecture
- Class-conditioned image generation
- MNIST digit generation with specified conditions
- Hyperparameter tuning and optimization

**Assignment:** Conditional digit generation on MNIST dataset
- **Task**: Generate specific digits (0-9) based on user input condition
- **Dataset**: MNIST handwritten digits dataset
- **Implementation Steps**:
  - Data preprocessing with normalization
  - cGAN architecture definition with initial hyperparameters
  - Extensive hyperparameter tuning for optimal performance
  - Model testing and validation
- **Objective**: Create controllable digit generation where users can specify which digit to generate

### Week 7: Pix2Pix - Paired Image Translation
- U-Net generator architecture
- PatchGAN discriminator
- L1 loss integration with adversarial loss
- Sketch-to-image translation on facades dataset

**Assignment:** Facades dataset image translation
- **Dataset**: Building facades with paired sketch-to-photo examples
- **Task**: Convert architectural sketches to realistic colored building images
- **Preprocessing Pipeline**:
  - Image jittering for data augmentation
  - Random cropping for varied perspectives
  - Normalization for stable training
- **Loss Functions**: Combined adversarial loss and L1 (Mean Absolute Error) loss
- **Objective**: High-quality image-to-image translation maintaining architectural details

### Week 8: CycleGANs - Unpaired Translation
- CycleGAN architecture with dual generators and discriminators
- Cycle consistency loss implementation
- Identity loss for color preservation
- Summer-to-winter landscape translation

**Assignment:** Seasonal landscape transformation
- **Dataset**: Unpaired summer and winter landscape images
- **Task**: Bidirectional translation between summer and winter scenes
- **Advanced Preprocessing**:
  - Image jittering for enhanced data diversity
  - Strategic cropping for optimal composition
  - Comprehensive normalization pipeline
- **Multi-Loss Training**:
  - Adversarial loss for realistic generation
  - Cycle consistency loss for content preservation
  - Identity loss for color composition maintenance
- **Objective**: High-quality seasonal transformation maintaining landscape structure while changing seasonal characteristics

## Technical Architecture

### CycleGAN Components
- **Two Generators**: G: X→Y and F: Y→X for bidirectional translation
- **Two Discriminators**: D_X and D_Y for domain-specific discrimination
- **Loss Functions**:
  - Adversarial Loss: Ensures realistic image generation
  - Cycle Consistency Loss: Maintains content preservation
  - Identity Loss: Preserves color composition

### Data Preprocessing
- Image jittering and random cropping
- Normalization to [-1, 1] range
- Data augmentation for improved generalization

## Results and Impact

### Project Outcomes
- **Successful Implementation**: Achieved high-quality unpaired image-to-image translation with CycleGAN architecture
- **Seasonal Transformation**: Successfully demonstrated bidirectional summer-to-winter landscape translation maintaining structural integrity while transforming seasonal characteristics
- **Technical Mastery**: Completed comprehensive progression from basic ML concepts to advanced generative models
- **Practical Skills**: Gained hands-on experience with multiple GAN architectures (DCGAN, cGAN, Pix2Pix, CycleGAN)

### Real-World Applications and Impact

**Creative Industries**
- **Digital Art and Design**: Transform artistic styles, convert sketches to paintings, enable cross-domain artistic expression
- **Film and Entertainment**: Season/weather modification in post-production, historical scene reconstruction, visual effects enhancement
- **Fashion and Retail**: Style transfer for clothing design, seasonal collection adaptation, virtual try-on experiences

**Medical and Healthcare**
- **Medical Imaging**: Cross-modality image translation (MRI to CT, X-ray enhancement), improving diagnostic accuracy
- **Pathology**: Stain normalization across different laboratory protocols, enhancing diagnostic consistency
- **Telemedicine**: Image quality enhancement for remote consultations, standardizing imaging across different devices

**Urban Planning and Architecture**
- **Smart City Development**: Seasonal visualization for urban planning, climate adaptation modeling
- **Real Estate**: Property visualization across seasons, architectural style transformation
- **Environmental Monitoring**: Landscape change detection, ecological impact assessment

**Autonomous Systems**
- **Self-Driving Cars**: Weather condition adaptation, improving perception across seasonal changes
- **Robotics**: Environment adaptation for robots operating in varying conditions
- **Surveillance Systems**: Enhanced visibility across different lighting and weather conditions

**E-commerce and Marketing**
- **Product Visualization**: Seasonal product adaptation, style customization for different markets
- **Virtual Staging**: Real estate property enhancement, furniture and decor style transformation
- **Brand Adaptation**: Logo and design adaptation across cultural contexts

**Scientific Research**
- **Climate Studies**: Visualizing climate change impacts, seasonal ecosystem modeling
- **Agricultural Technology**: Crop monitoring across seasons, pest detection enhancement
- **Remote Sensing**: Satellite image enhancement, cross-temporal analysis

### Technical Achievements
- **Unpaired Learning**: Eliminated the need for expensive paired datasets, making the technology accessible for diverse applications
- **Domain Adaptation**: Demonstrated robust performance across different image domains without domain-specific training
- **Quality Preservation**: Maintained image quality while performing complex transformations through advanced loss functions
- **Scalability**: Developed framework adaptable to various image translation tasks with minimal modifications

### Future Implications
The project establishes a foundation for advanced computer vision applications, contributing to the democratization of AI-powered image transformation technologies. The implementation serves as a stepping stone for more sophisticated generative models and real-world deployment scenarios.

## Technologies Used

- **Python**: Core programming language
- **TensorFlow/Keras**: Deep learning framework
- **NumPy**: Numerical computations
- **Pandas**: Data manipulation
- **OpenCV**: Image processing
- **Matplotlib**: Data visualization

## Project Leadership

**Mentors:** Abhishek Khandelwal, Vidhi Jain, Kushagra Singh

## References

- [Neural Networks from Scratch](https://towardsdatascience.com/math-neural-network-from-scratch-in-python-d6da9f29ce65)
- [CNN Architecture Guide](https://medium.com/codex/understanding-convolutional-neural-networks-a-beginners-journey-into-the-architecture-aab30dface10)
- [TensorFlow DCGAN Tutorial](https://www.tensorflow.org/tutorials/generative/dcgan)
- [TensorFlow CycleGAN Tutorial](https://www.tensorflow.org/tutorials/generative/cyclegan)

## Conclusion

This project provided a comprehensive journey through machine learning and deep learning fundamentals, progressing from basic concepts to advanced generative models. The implementation of CycleGANs demonstrates the power of unsupervised learning in image translation tasks, opening possibilities for numerous real-world applications in computer vision and creative AI.

---

*Developed under the IITK Consulting Group, Science and Technology Council, IIT Kanpur*
