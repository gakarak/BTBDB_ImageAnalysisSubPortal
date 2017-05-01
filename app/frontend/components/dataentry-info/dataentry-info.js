/**
 * Created by alexanderkalinovsky on 4/30/17.
 */
'use strict';

angular.module('dataentryInfo', ['ngMaterial','ngMessages'])
.component('dataentryInfo', {
    templateUrl: '/frontend/components/dataentry-info/dataentry-info.html',
    bindings: {
        cases: "<"
    },
    controller: function ($mdDialog, $http) {
        var self = this;
        self.$onInit = function () {
            self.cases = [];
            self.updateInfo();
        };
        //////////////
        self.updateInfo = function() {
            $http({
                method: "GET",
                url: "/db/cases-info/",
                data: {
                    procOnly: false
                }
            }).then(function (data) {
                // self.cases =
                try {
                    if(data.data.retcode === 0) {
                        self.cases = data.data.responce;
                    } else {
                        console.log('Server return error: ' + data.data.errorstr);
                    }
                } catch (err) {
                    console.log('Error : [' + err + ']');
                }
                console.log(data);
            }, function (error) {
                console.log("error: show up error message" + error);
            });
        }
    }
});
