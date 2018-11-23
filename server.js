var express = require('express');
var bodyParser = require('body-parser');
var app = express();
var path = require('path')
var bodyParser = require('body-parser'); 
app.use(bodyParser.urlencoded({ extended: false }));

app.use('/css',express.static('css'))
app.use('/fonts',express.static('fonts'))
app.use('/img',express.static('img'))
app.use('/js',express.static('js'))
app.use('/scss',express.static('scss'))



app.get('/', (request, response) =>  {
  response.sendFile(`${__dirname}/index.html`);
});

app.post('/enterInfo', (request, response) =>  {
  
  response.sendFile(`${__dirname}/enterInfo.html`);
});

app.post('/output', callName);

app.listen(2222, () => console.info('Application running on port 2222'));

function callName(req, res) {
  
  var info= req.body;
  var spawn = require("child_process").spawn;
  var process = spawn('python',["./prediction.py",info.gender,info.age,info.weight,info.height,info.bloodPressure,
                                info.glucose,info.skin,info.insulin,info.pregnant] );
  process.stdout.on('data', function(data) {
       

      res.send(`<html lang="zxx" class="no-js">

      <head>
        <!-- Mobile Specific Meta -->
        <meta name="viewport" content="width=device-width, initial-scale=1, shrink-to-fit=no">
        <!-- Favicon-->
        <link rel="shortcut icon" href="img/fav.png">
        <!-- Author Meta -->
        <meta name="author" content="codepixer">
        <!-- Meta Description -->
        <meta name="description" content="">
        <!-- Meta Keyword -->
        <meta name="keywords" content="">
        <!-- meta character set -->
        <meta charset="UTF-8">
        <!-- Site Title -->
        <title>About</title>
      
        <link href="https://fonts.googleapis.com/css?family=Playfair+Display:400,700,900|Roboto:300,400,500,700" rel="stylesheet">
        <!--
            CSS
            ============================================= -->
        <link rel="stylesheet" href="css/linearicons.css">
        <link rel="stylesheet" href="css/font-awesome.min.css">
        <link rel="stylesheet" href="css/bootstrap.css">
        <link rel="stylesheet" href="css/magnific-popup.css">
        <link rel="stylesheet" href="css/nice-select.css">
        <link rel="stylesheet" href="css/animate.min.css">
        <link rel="stylesheet" href="https://code.jquery.com/ui/1.12.1/themes/base/jquery-ui.css">
        <link rel="stylesheet" href="css/owl.carousel.css">
        <link rel="stylesheet" href="css/main.css">
      </head>
      
      <body>
      
        <!-- Start Header Area -->
        <header id="header">
          <div class="container">
            <div class="row align-items-center justify-content-between d-flex">
              <div id="logo">
                <a href="index.html"><img src="img/logo.png" alt="" title="" /></a>
              </div>
              <nav id="nav-menu-container">
              <ul class="nav-menu">
              <li class="menu-active"><a href="index.html">Home</a></li>
              <li><a href="about.html">Home</a></li>
              <li><a href="events.html">Prediction</a></li>
              <li class="menu-has-children"><a href="">Diabetes</a>
                <ul>
                  <li><a href="blog-home.html">Type 1</a></li>
                  <li><a href="blog-single.html">Type 2</a></li>
                </ul>
              </li>
              <li><a href="contact.html">Contact</a></li>
            </ul>
              </nav><!-- #nav-menu-container -->
            </div>
          </div>
        </header>
        <!-- Start Header Area -->
      
        <!-- Start Condition Area -->
        <section class="condition-area section-gap">
          <div class="container">
            <div class="row align-items-center justify-content-center">
              <div class="col-lg-6 col-md-8 col-sm-10">
                <div class="condition-left owl-carousel owl-condition">
                  <img class="img-fluid" src="img/condition/doctor1.jpg" alt="">
                  <img class="img-fluid" src="img/condition/insulin.jpg" alt="">
                </div>
              </div>
              <div class="offset-lg-1 col-lg-5">
                <div class="condition-right">
                  <h2>` +data.toString()+ `</h2>
                  <p>
                    Diabetes can lead to a buildup of sugars in the blood, which can increase the risk of dangerous complications, 
                    including stroke and heart disease. To reduce the risks of being diabetes, we have some recommendations for you
                  </p>
                  <ul>
                    <li>Cut Sugar and Refined Carbs From Your Diet</li>
                    <li>Work Out Regularly</li>
                    <li>Drink Water as Your Primary Beverage</li>
                    <li>Follow a Very-Low-Carb Diet</li>
                    <li><a href="https://www.healthline.com/nutrition/prevent-diabetes#section6">Click for More Information</a>
                    </li>
                  </ul>
                  
                </div>
              </div>
            </div>
          </div>
        </section>
        <!-- End Condition Area -->

        <!-- start footer Area -->
        <footer class="footer-area section-gap">
          <div class="container">
            <div class="row">
              <div class="col-lg-5 col-md-6 col-sm-6">
                <div class="single-footer-widget">
                  <h6>About Us</h6>
                  <p>
                    Lorem ipsum dolor sit amet, consectetur adipisicing elit, sed do eiusmod tempor incididunt ut labore dolore
                    magna aliqua.
                  </p>
                  <p class="footer-text"><!-- Link back to Colorlib can't be removed. Template is licensed under CC BY 3.0. -->
      Copyright &copy;<script>document.write(new Date().getFullYear());</script> All rights reserved | This template is made with <i class="fa fa-heart-o" aria-hidden="true"></i> by <a href="https://colorlib.com" target="_blank">Colorlib</a>
      <!-- Link back to Colorlib can't be removed. Template is licensed under CC BY 3.0. --></p>
                </div>
              </div>
              <div class="col-lg-5  col-md-6 col-sm-6">
                <div class="single-footer-widget">
                  <h6>Newsletter</h6>
                  <p>Stay update with our latest</p>
                  <div class="" id="mc_embed_signup">
                    <form target="_blank" novalidate="true" action="https://spondonit.us12.list-manage.com/subscribe/post?u=1462626880ade1ac87bd9c93a&amp;id=92a4423d01"
                     method="get" class="form-inline">
                      <input class="form-control" name="EMAIL" placeholder="Enter Email" onfocus="this.placeholder = ''" onblur="this.placeholder = 'Enter Email '"
                       required="" type="email">
                      <button class="click-btn btn btn-default"><i class="fa fa-long-arrow-right" aria-hidden="true"></i></button>
                      <div style="position: absolute; left: -5000px;">
                        <input name="b_36c4fd991d266f23781ded980_aefe40901a" tabindex="-1" value="" type="text">
                      </div>
      
                      <div class="info"></div>
                    </form>
                  </div>
                </div>
              </div>
              <div class="col-lg-2 col-md-6 col-sm-6 social-widget">
                <div class="single-footer-widget">
                  <h6>Follow Us</h6>
                  <p>Let us be social</p>
                  <div class="footer-social d-flex align-items-center">
                    <a href="#"><i class="fa fa-facebook"></i></a>
                    <a href="#"><i class="fa fa-twitter"></i></a>
                    <a href="#"><i class="fa fa-dribbble"></i></a>
                    <a href="#"><i class="fa fa-behance"></i></a>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </footer>
        <!-- End footer Area -->
      
        <script src="js/vendor/jquery-2.2.4.min.js"></script>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/popper.js/1.12.9/umd/popper.min.js" integrity="sha384-ApNbgh9B+Y1QKtv3Rn7W3mgPxhU9K/ScQsAP7hUibX39j7fakFPskvXusvfa0b4Q"
         crossorigin="anonymous"></script>
        <script src="js/vendor/bootstrap.min.js"></script>
        <script type="text/javascript" src="https://maps.googleapis.com/maps/api/js?key=AIzaSyBhOdIF3Y9382fqJYt5I_sswSrEw5eihAA"></script>
        <script src="https://code.jquery.com/ui/1.12.1/jquery-ui.js"></script>
        <script src="js/easing.min.js"></script>
        <script src="js/hoverIntent.js"></script>
        <script src="js/superfish.min.js"></script>
        <script src="js/jquery.ajaxchimp.min.js"></script>
        <script src="js/jquery.magnific-popup.min.js"></script>
        <script src="js/owl.carousel.min.js"></script>
        <script src="js/jquery.sticky.js"></script>
        <script src="js/jquery.nice-select.min.js"></script>
        <script src="js/waypoints.min.js"></script>
        <script src="js/jquery.counterup.min.js"></script>
        <script src="js/parallax.min.js"></script>
        <script src="js/mail-script.js"></script>
        <script src="js/main.js"></script>
      </body>
      
      </html>`)
  } )
}

