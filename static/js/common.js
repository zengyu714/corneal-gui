$(document).on('click', '#runButton', function () {
    alert("The btn Run was clicked.");
    // https://stackoverflow.com/questions/32288722/call-python-function-from-js
});

// $(document).on('click', '#repoButtons', function () {
$('#repoButtons .btn').on('click', function () {
    $('#checkedRepo').val($(this).find('input').val());
    $('#checkedRepoText').html($(this).find('input').val());
});