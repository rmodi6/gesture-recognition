//
//  main functions
//

var SHARK2 = SHARK2 || {};

//
//  set up the system
//
$(document).ready(function () {
    SHARK2.sandboxTest();

    $('#cvsMain')[0].width = 432;
    $('#cvsMain')[0].height = 137;

    SHARK2.context = $('#cvsMain')[0].getContext('2d');
    SHARK2.context.strokeStyle = "#df4b26";
    SHARK2.context.lineJoin = "round";
    SHARK2.context.lineWidth = 5;

    $('#cvsMain').on('mousedown', SHARK2.canvasStart);
    $('#cvsMain').on('touchstart', SHARK2.canvasStart);

    $('#cvsMain').on('mousemove', SHARK2.canvasMove);
    $('#cvsMain').on('touchmove', SHARK2.canvasMove);
    
    $('#cvsMain').on('mouseup', SHARK2.canvasStop);
    $('#cvsMain').on('touchend', SHARK2.canvasStop);
});

//
//  sandbox testing specific functions
//
SHARK2.sandboxTest = function () {};

//
//  handling mousedown on the main canvas
//
SHARK2.canvasStart = function (e) {
    SHARK2.coords = [];
    SHARK2.context.clearRect(0, 0, $('#cvsMain').width(), $('#cvsMain').height());
    SHARK2.context.beginPath();

    var rect = $('#cvsMain')[0].getBoundingClientRect();
    var x, y;
    
    // Handle both mouse and touch events
    if (e.touches) {
        x = e.touches[0].clientX - rect.left;
        y = e.touches[0].clientY - rect.top;
    } else {
        x = e.clientX - rect.left;
        y = e.clientY - rect.top;
    }
    
    SHARK2.context.moveTo(x, y);
    SHARK2.context.stroke();

    SHARK2.isDragging = true;

    SHARK2.coords.push({
        x: x,
        y: y
    });
    
    e.preventDefault();
};

//
//  handling mousemove on the main canvas
//
SHARK2.canvasMove = function (e) {
    if (!SHARK2.isDragging) return;

    var rect = $('#cvsMain')[0].getBoundingClientRect();
    var x, y;
    
    // Handle both mouse and touch events
    if (e.touches) {
        x = e.touches[0].clientX - rect.left;
        y = e.touches[0].clientY - rect.top;
    } else {
        x = e.clientX - rect.left;
        y = e.clientY - rect.top;
    }
    
    SHARK2.context.lineTo(x, y);
    SHARK2.context.moveTo(x, y);
    SHARK2.context.stroke();

    SHARK2.coords.push({
        x: x,
        y: y
    });
    
    e.preventDefault();
};

//
//  handling mouseup on the main canvas
//
SHARK2.canvasStop = function (e) {
    if (!SHARK2.isDragging) return;
    
    var rect = $('#cvsMain')[0].getBoundingClientRect();
    var x, y;
    
    // Handle both mouse and touch events
    // For touch events, use changedTouches (touches array is empty at end)
    if (e.changedTouches) {
        x = e.changedTouches[0].clientX - rect.left;
        y = e.changedTouches[0].clientY - rect.top;
    } else {
        x = e.clientX - rect.left;
        y = e.clientY - rect.top;
    }
    
    SHARK2.coords.push({
        x: x,
        y: y
    });

    console.log(SHARK2.coords);

    $.ajax({
        type: 'POST',
        url: '/shark2',
        data: JSON.stringify(SHARK2.coords),
        dataType: "json",
        success: function (result) {
            console.log(result);
            $('#divInfo').html(result['best_word'] + ' ' + result['elapsed_time'])
        },
        error: function (result) {
        }
    });

    SHARK2.isDragging = false;
    SHARK2.context.closePath();
    
    e.preventDefault();
};