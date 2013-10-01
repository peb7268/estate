#Using Composer and Packages in stand along projects.
The goal of this repo is to demonstrate how to use proper PHP techniques such as namespacing,
abstraction, and the like all within the context of implementing composer packages without a full fledged framework like laravel.

##How to use composer
1. Setup a new project
2. Install composer ( if not already installed )
3. In your project, do composer init and fill out the prompts
4. Do composer install, sometimes you have to do sudo.


This places the desired packages inside a vendor directory. Next step is to PSR-0 autoload the packages into your bootup file, in this case it's index.php.

##Namespaces
Notice on line 26 I use the namespace. Then in my class I just tell it to be part of that namespace. Composer's PSR-0 autoload property will then autoload it assuming my dir structure is right.