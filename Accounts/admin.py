from django.contrib import admin


from django.contrib.auth.admin import UserAdmin
from .models import Account

# Register your models here.
class AccountAdmin(UserAdmin):
    list_display = ('email', 'username', 'first_name', 'last_name', 'is_admin')
    search_fields = ('email', 'username')
    readonly_fields = ('date_joined', 'last_login')
    list_display_links = ('email', 'username')
    filter_horizontal = ()
    list_filter = ()
    fieldsets = ()
    ordering = ('date_joined',)
admin.site.register(Account, AccountAdmin)