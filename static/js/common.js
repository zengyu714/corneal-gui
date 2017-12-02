// Fill the value of an input block
// =======================================================
// $(document).on('click', '#repoButtons', function () {
$('#repoButtons .btn').on('click', function () {
    $('#checkedRepo').val($(this).find('input').val());
    $('#checkedRepoText').html($(this).find('input').val());
});
// =======================================================

// Close the alert message
// =======================================================
$(document).on('click', '.close', function () {
    $("#alreadyRunAlert").alert('close');
    // https://stackoverflow.com/questions/32288722/call-python-function-from-js
});

// Show the alert response of button
// =======================================================
// $(document).on('click', '#runButton', function () {
//     alert("The btn Run was clicked.");
//     // https://stackoverflow.com/questions/32288722/call-python-function-from-js
// });
// =======================================================


// Collapse to see the detail of inferred frames
// =======================================================
$(document).ready(function () {
    $("#frameDetail").click(function () {
        $(".frameCollapse").collapse('toggle');
    });

    $("#videoDetail").click(function () {
        $(".videoCollapse").collapse('toggle');
    });

    $("#bioParamDetail").click(function () {
        $(".bioParamCollapse").collapse('toggle');
    });

    // $(".btn-success").click(function(){
    //     $(".collapse").collapse('show');
    // });
    // $(".btn-warning").click(function(){
    //     $(".collapse").collapse('hide');
    // });
});
// =======================================================


$(document).ready(function () {
    var $divs = $(".barDiv").hide(),
        current = 0;
    $divs.eq(0).show();

    function showNext() {
        if (current < $divs.length - 1) {
            $divs.eq(current).delay(1000).fadeOut('fast', function () {
                current++;
                $divs.eq(current).fadeIn('fast');
                showNext();
            });
        }
    }

    showNext();
});