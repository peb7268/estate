/* 	This simple script is the client side implementation
*	for my first cross domain ajax proxy script. It's mostly
* 	just a simple excercise and reference for myself.
*/
jQuery(document).ready(function($) {
	$('form a').on('click', function(evt){
		evt.preventDefault();
		var site, method, url, address, citystatezip;
		var zws_id = 'X1-ZWz1bjj5gs9yx7_1zmzq';

		//site 	= $('input[name="site"]').val();
		method 	= $('input[name="method"]').val();
		method 	= (method.length > 0) ? method : 'GET';
		address = $('input[name="address"]').val().split(' ').join('+'); 			//2114+Bigelow+Ave
		citystatezip = encodeURIComponent($('input[name="citystatezip"]').val()); 	//Seattle%2C+WA ( I think %2C is a comma )
		site 	= 'http://www.zillow.com/webservice/GetSearchResults.htm?zws-id=' + zws_id + '&address=' + address + '&citystatezip=' + citystatezip;

		$.ajax('handler.php', {
			data: { 'site' : site,
					'method': (method.length > 0) ? method : 'GET'
			},
			type: method
		}).done(function(resp){
			var response 	= $.parseXML(resp);
			var listing 	= $(response).find('response results result');
			var zestimate 	= Number($(listing).find('zestimate amount').text());
			var lowRange 	= Number($(listing).find('zestimate valuationRange low').text());
			var highRange 	= Number($(listing).find('zestimate valuationRange high').text());

			var homeMap 	= listing.find('links mapthishome').text();
			var comps 		= listing.find('links comparables').text();

			$('#zestimate td:nth-child(2)', '#results').html(zestimate);
			$('#lowRange td:nth-child(2)', '#results').html(lowRange);
			$('#highRange td:nth-child(2)', '#results').html(highRange);
			$('#homeMap td:nth-child(2)', '#results').html(homeMap);
			$('#comps td:nth-child(2)', '#results').html(comps);

			$('#results').fadeIn();
		});
	});

});