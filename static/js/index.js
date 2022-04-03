window.HELP_IMPROVE_VIDEOJS = false;

var INTERP_BASE = "https://homes.cs.washington.edu/~kpar/nerfies/interpolation/stacked";
var NUM_INTERP_FRAMES = 240;

var interp_images = [];
function preloadInterpolationImages() {
  for (var i = 0; i < NUM_INTERP_FRAMES; i++) {
    var path = INTERP_BASE + '/' + String(i).padStart(6, '0') + '.jpg';
    interp_images[i] = new Image();
    interp_images[i].src = path;
  }
}

function setInterpolationImage(i) {
  var image = interp_images[i];
  image.ondragstart = function() { return false; };
  image.oncontextmenu = function() { return false; };
  $('#interpolation-image-wrapper').empty().append(image);
}


$(document).ready(function() {
    // Check for click events on the navbar burger icon
    $(".navbar-burger").click(function() {
      // Toggle the "is-active" class on both the "navbar-burger" and the "navbar-menu"
      $(".navbar-burger").toggleClass("is-active");
      $(".navbar-menu").toggleClass("is-active");

    });

    var options = {
			slidesToScroll: 1,
			slidesToShow: 3,
			loop: true,
			infinite: true,
			autoplay: false,
			autoplaySpeed: 3000,
    }

		// Initialize all div with carousel class
    var carousels = bulmaCarousel.attach('.carousel', options);

    // Loop on each carousel initialized
    for(var i = 0; i < carousels.length; i++) {
    	// Add listener to  event
    	carousels[i].on('before:show', state => {
    		console.log(state);
    	});
    }

    // Access to bulmaCarousel instance of an element
    var element = document.querySelector('#my-element');
    if (element && element.bulmaCarousel) {
    	// bulmaCarousel instance is available as element.bulmaCarousel
    	element.bulmaCarousel.on('before-show', function(state) {
    		console.log(state);
    	});
    }

    /*var player = document.getElementById('interpolation-video');
    player.addEventListener('loadedmetadata', function() {
      $('#interpolation-slider').on('input', function(event) {
        console.log(this.value, player.duration);
        player.currentTime = player.duration / 100 * this.value;
      })
    }, false);*/
    preloadInterpolationImages();

    $('#interpolation-slider').on('input', function(event) {
      setInterpolationImage(this.value);
    });
    setInterpolationImage(0);
    $('#interpolation-slider').prop('max', NUM_INTERP_FRAMES - 1);

    bulmaSlider.attach();

})

var currentCount = 0;
var meshLinks = [
  "static/meshes/mesh_sofa.glb",
  "static/meshes/mesh_desk.glb",
  "static/meshes/mesh_3m.glb",
  "static/meshes/mesh_lab.glb",
];
var imageLinks = [
  "https://raw.githubusercontent.com/cangumeli/ROCA/main/network/assets/sofa.jpg",
  "https://raw.githubusercontent.com/cangumeli/ROCA/main/network/assets/desk.jpg",
  "https://raw.githubusercontent.com/cangumeli/ROCA/main/network/assets/3m.jpg",
  "https://raw.githubusercontent.com/cangumeli/ROCA/main/network/assets/lab.jpg",
];

function switchToNext() {
  currentCount++;
  currentCount = currentCount % meshLinks.length;
  document.getElementById('inputImage').src = imageLinks[currentCount];
  document.getElementById('viewer3D').src = meshLinks[currentCount];
  document.getElementById('inputImageMobile').src = imageLinks[currentCount];
  document.getElementById('viewer3DMobile').src = meshLinks[currentCount];
}

function switchToPrev() {
  currentCount--;
  if (currentCount < 0) {
    currentCount += meshLinks.length;
  }
  document.getElementById('inputImage').src = imageLinks[currentCount];
  document.getElementById('viewer3D').src = meshLinks[currentCount];
  document.getElementById('inputImageMobile').src = imageLinks[currentCount];
  document.getElementById('viewer3DMobile').src = meshLinks[currentCount];
}

window.onload = () => {
  if( /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent) ) {
    // document.getElementById('videoSpan').hidden = true;
    /*document.getElementById('shortVideo').hidden = true;
    document.getElementById('shortVideoMobile').src = document.getElementById('shortVideo').src;
    document.getElementById('shortVideoMobile').hidden = false;*/
    document.getElementById('shortVideo').autoplay = false;
    document.getElementById('shortVideo').loop = false;
    document.getElementById('shortVideo').controls = false;

    document.getElementById('demoDesktop').hidden = true;
    document.getElementById('demoMobile').hidden = false;
    document.getElementById('userMessage').innerHTML = "The interactive mesh viewer should appear at the bottom";
  }
  // switchToNext();
}
